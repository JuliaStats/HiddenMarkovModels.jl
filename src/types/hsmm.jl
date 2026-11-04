"""
$(TYPEDEF)

Basic implementation of an HSMM with explicit state durations.

!!! warning
    Zero entries of a structured or sparse `trans` are only treated as impossible transitions (log-probability `-Inf`) if it is a `SparseMatrixCSC`. Other sparse or structured matrix types may give silently wrong results.

# Fields

$(TYPEDFIELDS)
"""
struct HSMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    VDur<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHSMM
    "initial state probabilities"
    init::V
    "state transition probabilities (zero diagonal, no self-transitions)"
    trans::M
    "observation distributions"
    dists::VD
    "state duration distributions, interpreted as the law of `(sojourn - 1)`"
    dur_dists::VDur
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function HSMM(
        init::AbstractVector,
        trans::AbstractMatrix,
        dists::AbstractVector,
        dur_dists::AbstractVector,
    )
        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans)
        hsmm = new{
            typeof(init),
            typeof(trans),
            typeof(dists),
            typeof(dur_dists),
            typeof(log_init),
            typeof(log_trans),
        }(
            init, trans, dists, dur_dists, log_init, log_trans
        )
        @argcheck valid_hsmm(hsmm)
        return hsmm
    end
end

function Base.show(io::IO, hsmm::HSMM)
    return print(
        io,
        "Hidden Semi-Markov Model with:\n - initialization: $(hsmm.init)\n - transition matrix: $(hsmm.trans)\n - observation distributions: [$(join(hsmm.dists, ", "))]\n - duration distributions: [$(join(hsmm.dur_dists, ", "))]",
    )
end

initialization(hsmm::HSMM) = hsmm.init
log_initialization(hsmm::HSMM) = hsmm.loginit
transition_matrix(hsmm::HSMM) = hsmm.trans
log_transition_matrix(hsmm::HSMM) = hsmm.logtrans
obs_distributions(hsmm::HSMM) = hsmm.dists
duration_distributions(hsmm::HSMM) = hsmm.dur_dists

## Fitting

#=
Sum the transition marginals `ξ[t1:t2-1]` of each sequence into `ξ[t2]`, which is free to use
as scratch space since no transition leaves the last timestep (its marginal is zero).
=#
function sum_transition_marginals!(
    ξ::AbstractVector{<:AbstractMatrix}, seq_ends::AbstractVectorOrNTuple{Int}
)
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            sum_transition_marginals!(ξ, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            sum_transition_marginals!(ξ, seq_ends, k)
        end
    end
    return nothing
end

function sum_transition_marginals!(
    ξ::AbstractVector{<:AbstractMatrix}, seq_ends::AbstractVectorOrNTuple{Int}, k::Integer
)
    t1, t2 = seq_limits(seq_ends, k)
    # An empty sequence has no slot of its own: `ξ[t2]` belongs to the previous sequence.
    t1 > t2 && return nothing
    scratch = ξ[t2]
    fill!(scratch, zero(eltype(scratch)))
    for t in t1:(t2 - 1)
        scratch .+= ξ[t]
    end
    return nothing
end

function first_nonempty_sequence(seq_ends::AbstractVectorOrNTuple{Int})
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        t1 <= t2 && return k
    end
    throw(ArgumentError("At least one sequence must be nonempty."))
end

#=
Re-estimate `trans` from the transition marginals, summed over the nonempty sequences in the
scratch slot `ξ[t2]` of the first one, by normalizing each row. A row without outgoing mass would
give `0 / 0`, so it keeps its previous values.
=#
function fit_transitions!(
    trans::AbstractMatrix,
    ξ::AbstractVector{<:AbstractMatrix},
    seq_ends::AbstractVectorOrNTuple{Int},
)
    k_first = first_nonempty_sequence(seq_ends)
    sum_transition_marginals!(ξ, seq_ends)
    _, t2_first = seq_limits(seq_ends, k_first)
    total = ξ[t2_first]
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        (k == k_first || t1 > t2) && continue
        total .+= ξ[t2]
    end
    normalize_transitions!(trans, total)
    return nothing
end

function normalize_transitions!(trans::AbstractMatrix, total::AbstractMatrix)
    for i in axes(trans, 1)
        outgoing = zero(eltype(total))
        for j in axes(trans, 2)
            outgoing += total[i, j]
        end
        iszero(outgoing) && continue
        for j in axes(trans, 2)
            trans[i, j] = total[i, j] / outgoing
        end
    end
    return nothing
end

#= Scalar indexing of a sparse matrix is a binary search, so work on the stored entries, which
`total` shares with `trans` when it was allocated with `similar`. =#
function normalize_transitions!(trans::SparseMatrixCSC, total::SparseMatrixCSC)
    if trans.colptr != total.colptr || rowvals(trans) != rowvals(total)
        return invoke(
            normalize_transitions!, Tuple{AbstractMatrix,AbstractMatrix}, trans, total
        )
    end
    rows, vals, total_vals = rowvals(trans), nonzeros(trans), nonzeros(total)
    outgoing = zeros(eltype(total), size(trans, 1))
    for p in eachindex(total_vals)
        outgoing[rows[p]] += total_vals[p]
    end
    for p in eachindex(vals)
        i = rows[p]
        iszero(outgoing[i]) || (vals[p] = total_vals[p] / outgoing[i])
    end
    return nothing
end

"""
$(SIGNATURES)

Re-estimate the parameters of `hsmm` from the posterior quantities in `fb_storage`, which must
come from [`forward_backward!`](@ref) with `transition_marginals=true` and
`duration_counts=true`. The duration distributions are refitted with
[`HiddenMarkovModels.fit_duration_in_sequence!`](@ref).

The parameters of a state without posterior weight are left unchanged.
"""
function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ, completed_durations, censored_durations) = fb_storage
    @argcheck has_transition_marginals(fb_storage) "The storage has no transition marginals, use `initialize_forward_backward` with `transition_marginals=true`."
    @argcheck has_duration_counts(fb_storage) "The storage has no duration counts, use `initialize_forward_backward` with `duration_counts=true`."
    @argcheck any(k -> <=(seq_limits(seq_ends, k)...), eachindex(seq_ends)) "At least one sequence must be nonempty."
    N = length(hsmm)
    # Fit states
    fill!(hsmm.init, zero(eltype(hsmm.init)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        t1 > t2 && continue  # empty sequence
        hsmm.init .+= view(γ, :, t1)
    end
    sum_to_one!(hsmm.init)
    fit_transitions!(hsmm.trans, ξ, seq_ends)
    # Fit observations
    for i in 1:N
        γᵢ = view(γ, i, :)
        iszero(sum(γᵢ)) && continue
        fit_in_sequence!(hsmm.dists, i, obs_seq, γᵢ)
    end
    #= Fit durations. With right-censored sojourns only, the durations have no upper bound and
    the maximizer runs off to an infinite mean, so the distribution is kept, which keeps the
    loglikelihood non-decreasing. =#
    for i in 1:N
        completed = view(completed_durations, :, i)
        iszero(sum(completed)) && continue
        censored = view(censored_durations, :, i)
        fit_duration_in_sequence!(hsmm.dur_dists, i, completed, censored)
    end
    # Update logs
    hsmm.loginit .= log.(hsmm.init)
    mynonzeros(hsmm.logtrans) .= log.(mynonzeros(hsmm.trans))
    # Safety check
    @argcheck valid_hsmm(hsmm)
    return nothing
end
