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

function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ, completed_durations, censored_durations) = fb_storage
    @argcheck has_duration_counts(fb_storage)
    # Fit states
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            t1 > t2 && continue  # an empty sequence has no ξ[t2] of its own
            scratch = ξ[t2]  # use ξ[t2] as scratch space since it is zero anyway
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            t1 > t2 && continue  # an empty sequence has no ξ[t2] of its own
            scratch = ξ[t2]  # use ξ[t2] as scratch space since it is zero anyway
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    end
    fill!(hsmm.init, zero(eltype(hsmm.init)))
    total = ξ[last(seq_ends)]  # the scratch space of the last nonempty sequence
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        t1 > t2 && continue
        hsmm.init .+= view(γ, :, t1)
        t2 < last(seq_ends) && (total .+= ξ[t2])
    end
    sum_to_one!(hsmm.init)
    # A state that is never left keeps its transitions instead of getting 0 / 0.
    for (trans_row, total_row) in zip(eachrow(hsmm.trans), eachrow(total))
        iszero(sum(total_row)) && continue
        trans_row .= total_row
        sum_to_one!(trans_row)
    end
    # Fit observations
    for i in 1:length(hsmm)
        fit_in_sequence!(hsmm.dists, i, obs_seq, view(γ, i, :))
    end
    # Fit durations, except for states never left, whose MLE has an infinite mean.
    for i in 1:length(hsmm)
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
