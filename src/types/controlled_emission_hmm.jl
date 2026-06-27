"""
$(TYPEDEF)

Abstract supertype for control-aware emission distributions used by
[`ControlledEmissionHMM`](@ref).

A concrete subtype `D <: ControlledEmission` describes an emission whose density,
sampling and fitting all depend on an external control value. It must implement the three
non-standard, control-aware methods:

- `DensityInterface.logdensityof(d::D, obs, control)` for inference
- `Random.rand(rng::AbstractRNG, d::D, control)` for sampling
- `StatsAPI.fit!(d::D, obs_seq, control_seq, weights)` for learning

Subtyping `ControlledEmission` makes these requirements explicit and provides
`DensityInterface.DensityKind(::D) = HasDensity()` automatically.
"""
abstract type ControlledEmission end

DensityInterface.DensityKind(::ControlledEmission) = HasDensity()

# Required interface fallbacks
function DensityInterface.logdensityof(d::ControlledEmission, obs, control)
    return throw(MethodError(logdensityof, (d, obs, control)))
end

function Random.rand(rng::AbstractRNG, d::ControlledEmission, control)
    return throw(MethodError(rand, (rng, d, control)))
end

function StatsAPI.fit!(d::ControlledEmission, obs_seq, control_seq, weights)
    return throw(MethodError(fit!, (d, obs_seq, control_seq, weights)))
end

"""
$(TYPEDEF)

A single [`ControlledEmission`](@ref) bound to a specific control value. Wraps a
control-aware emission together with one control, exposing the standard
`logdensityof(_, obs)` and `rand(rng, _)` interface expected by the inference algorithms.

$(TYPEDFIELDS)
"""
struct ControlBoundEmission{D<:ControlledEmission,C}
    "Control-aware emission distribution (a [`ControlledEmission`](@ref) subtype)."
    dist::D
    "The control value bound to this emission; passed as the third argument to
    `logdensityof` and `rand`."
    control::C
end

DensityInterface.DensityKind(::ControlBoundEmission) = HasDensity()
function DensityInterface.logdensityof(ce::ControlBoundEmission, obs)
    return logdensityof(ce.dist, obs, ce.control)
end
Random.rand(rng::AbstractRNG, ce::ControlBoundEmission) = rand(rng, ce.dist, ce.control)

"""
$(TYPEDEF)

A lazy vector pairing each [`ControlledEmission`](@ref) in `dists` with the same `control`
value, yielding [`ControlBoundEmission`](@ref) elements. No allocation occurs until
individual elements are accessed.

This is returned by `obs_distributions(hmm::ControlledEmissionHMM, control)` so that the
standard contract is satisfied: the returned distributions are already bound to `control`
and respond to `logdensityof(dist, obs)` and `rand(rng, dist)`.

$(TYPEDFIELDS)
"""
struct ControlBoundEmissionVector{D<:ControlledEmission,VD<:AbstractVector{D},C} <:
       AbstractVector{ControlBoundEmission{D,C}}
    "Vector of control-aware emission distributions ([`ControlledEmission`](@ref) subtypes)."
    dists::VD
    "The control value shared across all emissions; passed to each wrapped distribution."
    control::C
end

Base.size(ce::ControlBoundEmissionVector) = size(ce.dists)
function Base.eltype(::Type{ControlBoundEmissionVector{D,VD,C}}) where {D,VD,C}
    return ControlBoundEmission{D,C}
end
function Base.getindex(ce::ControlBoundEmissionVector, i::Int)
    return ControlBoundEmission(ce.dists[i], ce.control)
end

"""
$(TYPEDEF)

An [`AbstractHMM`](@ref) where control variables affect only the emission distributions,
not the transition dynamics. The `init` and `trans` fields are control-independent;
each element of `dists` must be a [`ControlledEmission`](@ref) subtype, supporting the
non-standard signatures `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`.

At inference time, `obs_distributions(hmm, control)` returns a lazy
[`ControlBoundEmissionVector`](@ref) that binds each emission to `control`, exposing the
standard `logdensityof(dist, obs)` / `rand(rng, dist)` interface expected by the inference
algorithms.

A `nothing` control is forwarded to the emissions like any other control value; for
genuinely uncontrolled models, prefer [`HMM`](@ref).

$(TYPEDFIELDS)
"""
struct ControlledEmissionHMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector{<:ControlledEmission},
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHMM
    "initial state probabilities"
    init::V
    "state transition probabilities (control-independent)"
    trans::M
    "control-aware emission distributions; each is a [`ControlledEmission`](@ref) subtype
    supporting `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`"
    dists::VD
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function ControlledEmissionHMM(
        init::AbstractVector,
        trans::AbstractMatrix,
        dists::AbstractVector{<:ControlledEmission},
    )
        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans)
        hmm = new{
            typeof(init),typeof(trans),typeof(dists),typeof(log_init),typeof(log_trans)
        }(
            init, trans, dists, log_init, log_trans
        )
        @argcheck valid_hmm(hmm)
        return hmm
    end
end

initialization(hmm::ControlledEmissionHMM) = hmm.init
log_initialization(hmm::ControlledEmissionHMM) = hmm.loginit

transition_matrix(hmm::ControlledEmissionHMM) = hmm.trans
log_transition_matrix(hmm::ControlledEmissionHMM) = hmm.logtrans

transition_matrix(hmm::ControlledEmissionHMM, ::Any) = hmm.trans
transition_matrix(hmm::ControlledEmissionHMM, ::Nothing) = hmm.trans
log_transition_matrix(hmm::ControlledEmissionHMM, ::Any) = hmm.logtrans
log_transition_matrix(hmm::ControlledEmissionHMM, ::Nothing) = hmm.logtrans

#=
valid_hmm cannot call obs_distributions(hmm, nothing) here because raw dists are
control-aware and don't implement the no-control DensityInterface. Only validate
structural consistency (lengths, init, trans).
=#
function valid_hmm(hmm::ControlledEmissionHMM)
    init = initialization(hmm)
    trans = transition_matrix(hmm)
    if !(length(init) == length(hmm.dists) == size(trans, 1) == size(trans, 2))
        return false
    elseif !valid_prob_vec(init)
        return false
    elseif !valid_trans_mat(trans)
        return false
    end
    return true
end

# Returns a lazy ControlBoundEmissionVector:
# each element is bound to `control` and responds to logdensityof(dist, obs) / rand(rng, dist)
function obs_distributions(hmm::ControlledEmissionHMM, control)
    return ControlBoundEmissionVector(hmm.dists, control)
end

function obs_distributions(hmm::ControlledEmissionHMM, control::Nothing)
    return ControlBoundEmissionVector(hmm.dists, control)
end
Base.length(hmm::ControlledEmissionHMM) = length(hmm.dists)

function StatsAPI.fit!(
    hmm::ControlledEmissionHMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ) = fb_storage

    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    end

    fill!(hmm.init, zero(eltype(hmm.init)))
    fill!(hmm.trans, zero(eltype(hmm.trans)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        hmm.trans .+= ξ[t2]
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))

    # Fit emissions with control, must use raw dists directly
    for i in 1:length(hmm)
        fit_in_sequence!(hmm.dists, i, obs_seq, control_seq, view(γ, i, :))
    end

    hmm.loginit .= log.(hmm.init)
    mynonzeros(hmm.logtrans) .= log.(mynonzeros(hmm.trans))

    @argcheck valid_hmm(hmm)
    return nothing
end

#=
`ControlledEmissionHMM` requires a control sequence: surface a `MethodError`
(instead of a downstream `ArgumentError`) so the hint registers in `__init__` fires.
=#
function Random.rand(rng::AbstractRNG, hmm::ControlledEmissionHMM, T::Integer)
    throw(MethodError(rand, (rng, hmm, T)))
end
function Random.rand(hmm::ControlledEmissionHMM, T::Integer)
    throw(MethodError(rand, (hmm, T)))
end

function _methoderror_haskw(kwargs, name::Symbol)
    if kwargs isa NamedTuple || kwargs isa Base.Pairs
        return haskey(kwargs, name)
    end
    return any(kwargs) do kw
        kw === name ||
            (kw isa Pair && first(kw) === name) ||
            (kw isa Tuple && !isempty(kw) && first(kw) === name)
    end
end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === rand &&
            isempty(kwargs) &&
            (
                (
                    length(argtypes) == 2 &&
                    argtypes[1] <: ControlledEmissionHMM &&
                    argtypes[2] <: Integer
                ) || (
                    length(argtypes) == 3 &&
                    argtypes[1] <: AbstractRNG &&
                    argtypes[2] <: ControlledEmissionHMM &&
                    argtypes[3] <: Integer
                )
            )
            print(
                io,
                "\nHint: `ControlledEmissionHMM` requires a control sequence. " *
                "Call `rand(hmm, control_seq)` instead.",
            )

        elseif exc.f === StatsAPI.fit! &&
            length(argtypes) == 3 &&
            argtypes[1] <: ControlledEmissionHMM &&
            argtypes[2] <: ForwardBackwardStorage &&
            argtypes[3] <: AbstractVector &&
            _methoderror_haskw(kwargs, :seq_ends) &&
            !_methoderror_haskw(kwargs, :control_seq)
            print(
                io,
                "\nHint: `ControlledEmissionHMM` requires `control_seq`. " *
                "Call `fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=...)` instead.",
            )
        end
    end
end
