"""
    AbstractHSMM

Abstract supertype for a Hidden Semi-Markov Model amenable to simulation, inference and learning.

An `AbstractHSMM` explicitly models state durations through a per-state duration distribution.
Sojourn lengths are drawn from those distributions, so self-transitions in the transition matrix
are forbidden (the diagonal must be zero).

[`AbstractHMM`](@ref) is a subtype of `AbstractHSMM`: every HMM is mathematically an HSMM whose
sojourn lengths follow a geometric distribution encoded by the diagonal of the transition matrix.
The interface defined here is inherited by `AbstractHMM`.

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
"""
abstract type AbstractHSMM end

@inline DensityInterface.DensityKind(::AbstractHSMM) = HasDensity()

## Interface

"""
    initialization(model)

Return the vector of initial state probabilities for `model` (any `AbstractHSMM`, including
`AbstractHMM` subtypes).
"""
function initialization end

"""
    transition_matrix(model)
    transition_matrix(model, control)

Return the matrix of state transition probabilities for `model` (possibly when `control` is applied).

For an [`AbstractHSMM`](@ref) the diagonal of this matrix must be zero (no self-transitions).

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t-1` to `t` (since version 0.7 of the package).
"""
function transition_matrix end

"""
    obs_distributions(model)
    obs_distributions(model, control)

Return a vector of observation distributions, one for each state of `model` (possibly when `control` is applied).

These distribution objects should implement

- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning
"""
function obs_distributions end

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
"""
function duration_distributions end

"""
    length(model)

Return the number of states of `model` (any `AbstractHSMM`, including `AbstractHMM`).
"""
Base.length(model::AbstractHSMM) = length(initialization(model))

"""
    eltype(model, obs, control)

Return a type that can accommodate forward-backward computations for `model` on observations similar to `obs`.

It is typically a promotion between the element type of the initialization, the element type of the transition matrix, and the type of an observation logdensity evaluated at `obs`.
"""
function Base.eltype(model::AbstractHSMM, obs, control)
    init_type = eltype(initialization(model))
    trans_type = eltype(transition_matrix(model, control))
    dist = obs_distributions(model, control)[1]
    logdensity_type = typeof(logdensityof(dist, obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

"""
    log_initialization(model)

Return the vector of initial state log-probabilities for `model`.

Falls back on `initialization`.
"""
log_initialization(model::AbstractHSMM) = elementwise_log(initialization(model))

"""
    log_transition_matrix(model)
    log_transition_matrix(model, control)

Return the matrix of state transition log-probabilities for `model` (possibly when `control` is applied).

Falls back on `transition_matrix`.

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t-1` to `t` (since version 0.7 of the package).
"""
log_transition_matrix(model::AbstractHSMM) = elementwise_log(transition_matrix(model))

function log_transition_matrix(model::AbstractHSMM, control)
    return elementwise_log(transition_matrix(model, control))
end

## Fallbacks for no control

transition_matrix(model::AbstractHSMM, ::Nothing) = transition_matrix(model)
log_transition_matrix(model::AbstractHSMM, ::Nothing) = log_transition_matrix(model)
obs_distributions(model::AbstractHSMM, ::Nothing) = obs_distributions(model)
duration_distributions(model::AbstractHSMM, ::Nothing) = duration_distributions(model)

## Prior

"""
    logdensityof(model)

Return the prior loglikelihood associated with the parameters of `model`.
"""
DensityInterface.logdensityof(model::AbstractHSMM) = false

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

function Random.rand(hsmm::AbstractHSMM, control_seq::AbstractVector)
    return rand(default_rng(), hsmm, control_seq)
end

function Random.rand(rng::AbstractRNG, hsmm::AbstractHSMM, T::Integer)
    return rand(rng, hsmm, Fill(nothing, T))
end

function Random.rand(hsmm::AbstractHSMM, T::Integer)
    return rand(hsmm, Fill(nothing, T))
end
