"""
    AbstractLatentStateModel

Abstract supertype for a discrete-latent-state sequence model amenable to simulation, inference
and learning.

This type factors out the methods shared by [`AbstractHMM`](@ref) and [`AbstractHSMM`](@ref)
(state initialization, transition matrix, observation distributions, and the derived `length`,
`eltype`, log-accessors and no-control fallbacks). 
# Interface

Every subtype must implement:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`fit!`](@ref) (for learning)

Subtypes may add further requirements (e.g. [`AbstractHSMM`](@ref) additionally requires
[`duration_distributions`](@ref)).
"""
abstract type AbstractLatentStateModel end

@inline DensityInterface.DensityKind(::AbstractLatentStateModel) = HasDensity()

## Interface

"""
    initialization(model)

Return the vector of initial state probabilities for `model` (any `AbstractLatentStateModel`).
"""
function initialization end

"""
    transition_matrix(model)
    transition_matrix(model, control)

Return the matrix of state transition probabilities for `model` (possibly when `control` is applied).


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
    length(model)

Return the number of states of `model` (any `AbstractLatentStateModel`).
"""
Base.length(model::AbstractLatentStateModel) = length(initialization(model))

"""
    eltype(model, obs, control)

Return a type that can accommodate forward-backward computations for `model` on observations similar to `obs`.

It is typically a promotion between the element type of the initialization, the element type of the transition matrix, and the type of an observation logdensity evaluated at `obs`. Subtypes that carry additional numeric ingredients extend this promotion (e.g. [`AbstractHSMM`](@ref) folds in the duration logdensity type).
"""
function Base.eltype(model::AbstractLatentStateModel, obs, control)
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
log_initialization(model::AbstractLatentStateModel) = elementwise_log(initialization(model))

"""
    log_transition_matrix(model)
    log_transition_matrix(model, control)

Return the matrix of state transition log-probabilities for `model` (possibly when `control` is applied).

Falls back on `transition_matrix`.

!!! note
    When processing sequences, the control at time `t` influences the transition from time `t-1` to `t` (since version 0.7 of the package).
"""
log_transition_matrix(model::AbstractLatentStateModel) =
    elementwise_log(transition_matrix(model))

function log_transition_matrix(model::AbstractLatentStateModel, control)
    return elementwise_log(transition_matrix(model, control))
end

## Fallbacks for no control

transition_matrix(model::AbstractLatentStateModel, ::Nothing) = transition_matrix(model)
function log_transition_matrix(model::AbstractLatentStateModel, ::Nothing)
    return log_transition_matrix(model)
end
obs_distributions(model::AbstractLatentStateModel, ::Nothing) = obs_distributions(model)

## Prior

"""
    logdensityof(model)

Return the prior loglikelihood associated with the parameters of `model`.
"""
DensityInterface.logdensityof(model::AbstractLatentStateModel) = false

## Fill logdensities

function obs_logdensities!(
    logb::AbstractVector{T},
    model::AbstractLatentStateModel,
    obs,
    control;
    error_if_not_finite::Bool=true,
) where {T}
    dists = obs_distributions(model, control)
    @simd for i in eachindex(logb, dists)
        logb[i] = logdensityof(dists[i], obs)
    end
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

## Sampling entry points shared across latent-state models

function Random.rand(model::AbstractLatentStateModel, control_seq::AbstractVector)
    return rand(default_rng(), model, control_seq)
end

function Random.rand(rng::AbstractRNG, model::AbstractLatentStateModel, T::Integer)
    return rand(rng, model, Fill(nothing, T))
end

function Random.rand(model::AbstractLatentStateModel, T::Integer)
    return rand(model, Fill(nothing, T))
end
