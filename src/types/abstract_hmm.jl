"""
    AbstractHMM

Abstract supertype for an HMM amenable to simulation, inference and learning.

`AbstractHMM` is a subtype of [`AbstractHSMM`](@ref) purely for code reuse: the shared interface
(`initialization`, `transition_matrix`, `obs_distributions`, `length`, `eltype`, no-control
fallbacks, etc.) is inherited from [`AbstractHSMM`](@ref). An HMM encodes its sojourn behavior
implicitly through the transition-matrix diagonal and does **not** implement
[`duration_distributions`](@ref); regular HMM modeling, inference and learning are unaffected by
the HSMM machinery.

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

## Log accessors

"""
    log_initialization(hmm)

Return the vector of initial state log-probabilities for `hmm`.

Falls back on `initialization`.
"""
log_initialization(hmm::AbstractHMM) = elementwise_log(initialization(hmm))

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

Return a named tuple `(; state_seq, obs_seq)`. This is more specific than the [`AbstractHSMM`](@ref)
`rand` (which additionally returns a `duration_seq`), preserving the HMM's historical return shape.
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

# The `rand(hmm, control_seq)`, `rand(rng, hmm, T)`, and `rand(hmm, T)` wrappers are inherited from
# `AbstractHSMM`. The inner `rand(rng, hmm, control_seq)` above is more specific than the
# `AbstractHSMM` version, so those wrappers ultimately dispatch to it, preserving HMM's 2-tuple
# `(state_seq, obs_seq)` return shape.
