"""
    AbstractHMM

Abstract supertype for an HMM amenable to simulation, inference and learning.

`AbstractHMM` is a subtype of [`AbstractHSMM`](@ref): every HMM is mathematically an HSMM whose
state sojourn lengths follow a geometric distribution with parameter `1 - a_{ii}` (where `a_{ii}`
is the self-transition probability). The shared interface (`initialization`, `transition_matrix`,
`obs_distributions`, `length`, `eltype`, no-control fallbacks, etc.) is inherited from
`AbstractHSMM`. Inference functions ([`forward`](@ref), [`viterbi`](@ref),
[`forward_backward`](@ref), [`baum_welch`](@ref), [`logdensityof`](@ref)) are then specialized on
`AbstractHMM` and override the segment-DP versions defined for `AbstractHSMM`, so HMM workloads
keep the fast scaled linear-space algorithms.

# Interface

To create your own subtype of `AbstractHMM`, you need to implement the following methods:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`fit!`](@ref) (for learning)

# Applicable functions

Any `AbstractHMM` which satisfies the interface can be given to the following functions:

- [`rand`](@ref)
- [`logdensityof`](@ref)
- [`forward`](@ref)
- [`viterbi`](@ref)
- [`forward_backward`](@ref)
- [`baum_welch`](@ref) (if `[fit!](@ref)` is implemented)
"""
abstract type AbstractHMM <: AbstractHSMM end

"""
    duration_distributions(hmm::AbstractHMM)
    duration_distributions(hmm::AbstractHMM, control)

Return the implied sojourn distributions of `hmm` viewed as an [`AbstractHSMM`](@ref): a vector of
[`GeometricDuration`](@ref) instances with `p = 1 - a_{ii}`, where `a_{ii}` are the diagonal
entries of [`transition_matrix`](@ref).

This default exists so that `AbstractHMM <: AbstractHSMM` is satisfied at the interface level; it
is not called on the hot path because HMM-specific inference methods are more specific and dispatch
ahead of the HSMM segment-DP code.
"""
function duration_distributions(hmm::AbstractHMM)
    trans = transition_matrix(hmm)
    return [GeometricDuration(1 - trans[i, i]) for i in 1:length(hmm)]
end

function duration_distributions(hmm::AbstractHMM, control)
    trans = transition_matrix(hmm, control)
    return [GeometricDuration(1 - trans[i, i]) for i in 1:length(hmm)]
end

# Disambiguate against `duration_distributions(::AbstractHSMM, ::Nothing)`.
duration_distributions(hmm::AbstractHMM, ::Nothing) = duration_distributions(hmm)

"""
    StatsAPI.fit!(
        hmm, fb_storage::ForwardBackwardStorage,
        obs_seq, [control_seq]; seq_ends,
    )

Update `hmm` in-place based on information generated during forward-backward.

This function is allowed to reuse `fb_storage` as a scratch space, so its contents should not be trusted afterwards.
"""
StatsAPI.fit!

## Fill logdensities

function obs_logdensities!(
    logb::AbstractVector{T}, hmm::AbstractHMM, obs, control; error_if_not_finite::Bool=true
) where {T}
    dists = obs_distributions(hmm, control)
    @simd for i in eachindex(logb, dists)
        logb[i] = logdensityof(dists[i], obs)
    end
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

## Sampling

"""
    rand([rng,] hmm, T)
    rand([rng,] hmm, control_seq)

Simulate `hmm` for `T` time steps, or when the sequence `control_seq` is applied.

Return a named tuple `(; state_seq, obs_seq)`. Note that this overrides the HSMM `rand`
(which additionally returns a `duration_seq`), preserving the HMM's historical return shape.
"""
function Random.rand(rng::AbstractRNG, hmm::AbstractHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    @views for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t + 1])
        state_seq[t + 1] = rand(
            rng, LightCategorical(trans[state_seq[t], :], dummy_log_probas)
        )
    end

    dists1 = obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = obs_distributions(hmm, control_seq[t])
        obs_seq[t] = rand(rng, dists[state_seq[t]])
    end
    return (; state_seq=state_seq, obs_seq=obs_seq)
end

# The `rand(hmm, control_seq)`, `rand(rng, hmm, T)`, and `rand(hmm, T)` wrappers are inherited
# from `AbstractHSMM`. The inner `rand(rng, hmm, control_seq)` defined above is more specific
# than the `AbstractHSMM` version and is what those wrappers ultimately dispatch to, preserving
# HMM's 2-tuple `(state_seq, obs_seq)` return shape.
