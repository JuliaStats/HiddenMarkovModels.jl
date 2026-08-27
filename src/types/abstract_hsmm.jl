"""
    AbstractHSMM

Abstract supertype for a Hidden Semi-Markov Model amenable to simulation, inference and learning.

An `AbstractHSMM` explicitly models state durations through a per-state duration distribution.
Sojourn lengths are drawn from those distributions, so self-transitions in the transition matrix
are forbidden (the diagonal must be zero).

# Interface

To create your own subtype of `AbstractHSMM`, you need to implement the following methods:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`duration_distributions`](@ref)
- [`fit!`](@ref) (for learning)

# Applicable functions

Any `AbstractHSMM` which satisfies the interface can be given to the following functions:

- [`rand`](@ref)
- [`logdensityof`](@ref)
- [`joint_logdensityof`](@ref)
- [`forward`](@ref)

"""
abstract type AbstractHSMM <: AbstractLatentStateModel end

## Interface

"""
    duration_distributions(model)
    duration_distributions(model, control)

Return a vector of sojourn-time distributions, one for each state of `model` (possibly when
`control` is applied).

Following the convention of [`duration_logdensityof`](@ref) and [`rand_duration`](@ref), each
distribution has support on `{0, 1, 2, ...}` and is interpreted as the law of `(sojourn time - 1)`.
Any density-bearing object works (e.g. a `Distributions.Geometric` or `Distributions.Poisson`); it
should implement

- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, k)` for inference
- [`duration_logsurvival`](@ref)`(dist, k)` for right-censoring the final segment of a sequence

The last method is already provided for every `Distributions.DiscreteUnivariateDistribution`.
"""
function duration_distributions end

duration_distributions(model::AbstractHSMM, ::Nothing) = duration_distributions(model)

#=
We split out the type handling for duration logdensities so that the shared `eltype` promotion
defined on `AbstractLatentStateModel` can be reused by both HMMs and HSMMs, with only the
HSMM branch folding in the duration term.
=#
function duration_logdensity_type(model::AbstractHSMM, control)
    dist = duration_distributions(model, control)[1]
    return typeof(duration_logdensityof(dist, 1))
end

#=
Extend the shared `eltype` (which promotes the initialization, transition and observation
logdensity types) with the duration logdensity type, since HSMM forward-backward also accumulates
duration terms.
=#
function Base.eltype(hsmm::AbstractHSMM, obs, control)
    base_type = @invoke Base.eltype(hsmm::AbstractLatentStateModel, obs, control)
    return promote_type(base_type, duration_logdensity_type(hsmm, control))
end

## Sampling

"""
    rand([rng,] hsmm::AbstractHSMM, T)
    rand([rng,] hsmm::AbstractHSMM, control_seq)

Simulate `hsmm` for `T` time steps, or when the sequence `control_seq` is applied.

Sojourn lengths are drawn from the model's duration distributions (via [`rand_duration`](@ref)), so
consecutive timesteps share the same state until the sojourn elapses, after which a new state is
drawn under the no-self-transition constraint.

Return a named tuple `(; state_seq, obs_seq, duration_seq)`, where `duration_seq[t]` is the sojourn
length originally sampled for the segment containing `t` (it may exceed the remaining horizon if the
sequence was truncated at `T`).
"""
function Random.rand(rng::AbstractRNG, hsmm::AbstractHSMM, control_seq::AbstractVector)
    T = length(control_seq)
    N = length(hsmm)

    init = initialization(hsmm)
    dummy_log_probas = fill(-Inf, N)
    current_state = rand(rng, LightCategorical(init, dummy_log_probas))

    obs_dists1 = obs_distributions(hsmm, control_seq[1])
    first_obs = rand(rng, obs_dists1[current_state])

    state_seq = Vector{Int}(undef, T)
    obs_seq = Vector{typeof(first_obs)}(undef, T)
    duration_seq = Vector{Int}(undef, T)

    t = 1
    while t <= T
        duration_dists = duration_distributions(hsmm, control_seq[t])
        current_duration = rand_duration(rng, duration_dists[current_state])

        end_time = min(t + current_duration - 1, T)
        for τ in t:end_time
            state_seq[τ] = current_state
            duration_seq[τ] = current_duration
            if τ == 1
                obs_seq[τ] = first_obs
            else
                obs_dists = obs_distributions(hsmm, control_seq[τ])
                obs_seq[τ] = rand(rng, obs_dists[current_state])
            end
        end

        t = end_time + 1

        if t <= T
            # The control at time `t` drives the transition into time `t`.
            trans = transition_matrix(hsmm, control_seq[t])
            current_state = rand(
                rng, LightCategorical(trans[current_state, :], dummy_log_probas)
            )
        end
    end

    return (; state_seq=state_seq, obs_seq=obs_seq, duration_seq=duration_seq)
end
