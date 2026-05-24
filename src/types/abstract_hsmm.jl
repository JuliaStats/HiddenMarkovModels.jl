"""
    AbstractHSMM

Abstract supertype for a Hidden Semi-Markov Model amenable to simulation, inference and learning.

An `AbstractHSMM` explicitly models state durations through a per-state
[`AbstractDurationDistribution`](@ref). Sojourn lengths are drawn from those distributions, so
self-transitions in the transition matrix are forbidden.

[`AbstractHMM`](@ref) is a subtype of `AbstractHSMM`: every HMM is mathematically an HSMM with
geometric sojourn distributions whose parameters are encoded by the diagonal of the transition
matrix. Most of the interface defined here is inherited by `AbstractHMM`; the segment-DP inference
methods defined for `AbstractHSMM` are then *overridden* by faster HMM-specific versions via
dispatch on the more specific type.

# Interface

To create your own subtype of `AbstractHSMM`, you need to implement:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`duration_distributions`](@ref)
- [`fit!`](@ref) (for learning)

# Applicable functions

Any `AbstractHSMM` which satisfies the interface can be given to the following functions:

- [`rand`](@ref)
- [`logdensityof`](@ref)
- [`forward`](@ref)
- [`viterbi`](@ref)
- [`forward_backward`](@ref)
- [`baum_welch`](@ref) (if `fit!` is implemented)
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

For an [`AbstractHSMM`](@ref) the diagonal of this matrix must be zero (no self-transitions); the
[`HSMM`](@ref) constructor enforces this on construction.

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t-1`
    to `t` (since version 0.7 of the package).
"""
function transition_matrix end

"""
    obs_distributions(model)
    obs_distributions(model, control)

Return a vector of observation distributions, one for each state of `model` (possibly when
`control` is applied).

These distribution objects should implement

- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning
"""
function obs_distributions end

"""
    duration_distributions(hsmm)
    duration_distributions(hsmm, control)

Return a vector of duration distributions, one for each state of `hsmm` (possibly when `control` is applied).

Each duration distribution should implement:

- `Random.rand(rng, dist)` for sampling positive-integer durations
- `DensityInterface.logdensityof(dist, k::Int)` for `log P(D = k)`
- `StatsAPI.fit!(dist, durations, weights)` for learning

See [`AbstractDurationDistribution`](@ref). For an [`AbstractHMM`](@ref), a default implementation
derives geometric durations from the diagonal of [`transition_matrix`](@ref).
"""
function duration_distributions end

"""
    length(model)

Return the number of states of `model` (any `AbstractHSMM`, including `AbstractHMM`).
"""
Base.length(model::AbstractHSMM) = length(initialization(model))

"""
    eltype(model, obs, control)

Return a type that can accommodate forward-backward computations for `model` on observations
similar to `obs`.

It is typically a promotion between the element type of the initialization, the element type of
the transition matrix, and the type of an observation logdensity evaluated at `obs`.
"""
function Base.eltype(model::AbstractHSMM, obs, control)
    init_type = eltype(initialization(model))
    trans_type = eltype(transition_matrix(model, control))
    dist = obs_distributions(model, control)[1]
    logdensity_type = typeof(logdensityof(dist, obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

## Fallbacks for no control

transition_matrix(hsmm::AbstractHSMM, ::Nothing) = transition_matrix(hsmm)
log_transition_matrix(hsmm::AbstractHSMM, ::Nothing) = log_transition_matrix(hsmm)
obs_distributions(hsmm::AbstractHSMM, ::Nothing) = obs_distributions(hsmm)
duration_distributions(hsmm::AbstractHSMM, ::Nothing) = duration_distributions(hsmm)

"""
    log_initialization(model)

Return the vector of initial state log-probabilities for `model`.

Default implementation calls `log.(initialization(model))`; override on a concrete subtype
that already stores log-initialization to avoid recomputation.
"""
log_initialization(hsmm::AbstractHSMM) = elementwise_log(initialization(hsmm))

"""
    log_transition_matrix(model)
    log_transition_matrix(model, control)

Return the matrix of state transition log-probabilities for `model` (possibly when `control`
is applied).

Default implementation calls `log.(transition_matrix(model, control))`; override on a concrete
subtype that already stores log-transitions to avoid recomputation.

!!! note
    When processing sequences, the control at time `t` influences the transition from time
    `t-1` to `t` (since version 0.7 of the package).
"""
log_transition_matrix(hsmm::AbstractHSMM) = elementwise_log(transition_matrix(hsmm))

function log_transition_matrix(hsmm::AbstractHSMM, control)
    return elementwise_log(transition_matrix(hsmm, control))
end

## Prior

DensityInterface.logdensityof(hsmm::AbstractHSMM) = false

## Sampling

"""
    rand([rng,] hsmm::AbstractHSMM, T)
    rand([rng,] hsmm::AbstractHSMM, control_seq)

Simulate `hsmm` for `T` time steps, or when the sequence `control_seq` is applied.

Sojourn lengths are drawn from the model's duration distributions, so consecutive timesteps share the
same state until the duration elapses (after which a new state is drawn under the no-self-transition
constraint).

Return a named tuple `(; state_seq, obs_seq, duration_seq)`, where `duration_seq[t]` is the duration
originally sampled for the sojourn containing `t` (it may exceed the actual sojourn length if the
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
        current_duration = rand(rng, duration_dists[current_state])

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
            trans = transition_matrix(hsmm, control_seq[t - 1])
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
