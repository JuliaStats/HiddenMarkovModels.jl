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

"""
$(SIGNATURES)

Re-estimate the parameters of `hsmm` from the posterior quantities in `fb_storage`, which must
come from [`forward_backward!`](@ref) with `transition_marginals=true` and
`duration_counts=true`. Empty sequences are skipped. The transition marginals `fb_storage.ξ` are
summed in place, so `forward_backward!` must run again before the next call.

The duration distributions are refitted by
[`HiddenMarkovModels.fit_duration_in_sequence!`](@ref), which accounts exactly for the
right-censored final sojourns.

The parameters of a state without posterior weight carry no information, so they are left
unchanged: the transitions out of a state that is never left, the observation distribution of a
state that is never visited, and the duration distribution of a state without completed
sojourns. In the last case, which includes a state only ever seen in right-censored final
sojourns, the data give no upper bound on its durations: the maximizer of the duration
objective runs off to an infinite mean, while keeping the distribution keeps the objective, and
hence the loglikelihood, non-decreasing. This is the only place where that policy is applied, so
methods of `fit_duration_in_sequence!` may assume completed sojourns.
"""
function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ, completed_durations, censored_durations) = fb_storage
    @argcheck fb_storage.filled_transition_marginals[] "Transition marginals are missing, call `forward_backward!` with `transition_marginals=true` first."
    @argcheck fb_storage.filled_duration_counts[] "Expected duration counts are missing, call `forward_backward!` with `duration_counts=true` first."
    N = length(hsmm)
    # Fit states (transitions first, since they check that some sequence is nonempty)
    fit_transitions!(hsmm.trans, ξ, seq_ends)
    # The transition marginals were summed in place.
    fb_storage.filled_transition_marginals[] = false
    fill!(hsmm.init, zero(eltype(hsmm.init)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        t1 > t2 && continue
        hsmm.init .+= view(γ, :, t1)
    end
    sum_to_one!(hsmm.init)
    # Fit observations
    for i in 1:N
        γᵢ = view(γ, i, :)
        iszero(sum(γᵢ)) && continue
        fit_in_sequence!(hsmm.dists, i, obs_seq, γᵢ)
    end
    # Fit durations, keeping the distribution of a state without completed sojourns.
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
