"""
$(TYPEDEF)

A single emission distribution bound to a specific control value.
Wraps a control-aware distribution and a control value, exposing
the standard `logdensityof(ce, obs)` and `rand(rng, ce)` interface.

$(TYPEDFIELDS)
"""
struct ControlledEmission{D,C}
    "Control-aware emission distribution. Must support the non-standard signatures
    `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`."
    dist::D
    "The control value bound to this emission; passed as the third argument to
    `logdensityof` and `rand`."
    control::C
end

DensityInterface.DensityKind(::ControlledEmission) = HasDensity()
function DensityInterface.logdensityof(ce::ControlledEmission, obs)
    logdensityof(ce.dist, obs, ce.control)
end
Random.rand(rng::AbstractRNG, ce::ControlledEmission) = rand(rng, ce.dist, ce.control)

"""
$(TYPEDEF)

A lazy vector of [`ControlledEmission`](@ref) wrappers, pairing each distribution in `dists`
with the same `control` value. No allocation occurs until individual elements are accessed.

This is returned by `obs_distributions(hmm::ControlledEmissionHMM, control::C)` so that the
standard `obs_distributions(hmm, control)` contract is satisfied: the returned distributions
are already bound to `control` and respond to `logdensityof(dist, obs)` and `rand(rng, dist)`.

$(TYPEDFIELDS)
"""
struct ControlledEmissions{D,VD<:AbstractVector{D},C} <:
       AbstractVector{ControlledEmission{D,C}}
    "Vector of control-aware emission distributions. Each element must support
    `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`."
    dists::VD
    "The control value shared across all emissions; passed to each wrapped distribution."
    control::C
end

Base.size(ce::ControlledEmissions) = size(ce.dists)
Base.eltype(::Type{ControlledEmissions{D,VD,C}}) where {D,VD,C} = ControlledEmission{D,C}
Base.getindex(ce::ControlledEmissions, i::Int) = ControlledEmission(ce.dists[i], ce.control)

"""
$(TYPEDEF)

An [`AbstractHMM`](@ref) where control variables affect only the emission distributions,
not the transition dynamics. The `init` and `trans` fields are control-independent;
each element of `dists` must be a control-aware distribution supporting the non-standard
signatures `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`.

At inference time, `obs_distributions(hmm, control)` returns a lazy
[`ControlledEmissions`](@ref) vector that binds each raw distribution to `control`,
exposing the standard `logdensityof(dist, obs)` / `rand(rng, dist)` interface expected
by the inference algorithms.

`nothing` is not a valid control value; use [`HMM`](@ref) for uncontrolled models.

$(TYPEDFIELDS)
"""
struct ControlledEmissionHMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHMM
    "initial state probabilities"
    init::V
    "state transition probabilities (control-independent)"
    trans::M
    "control-aware emission distributions; each element must support
    `logdensityof(dist, obs, control)` and `rand(rng, dist, control)`"
    dists::VD
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function ControlledEmissionHMM(
        init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector
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

# Returns raw meta-distributions (control-aware dist objects), used internally for fitting
obs_distributions(hmm::ControlledEmissionHMM) = hmm.dists

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

# ControlledEmissionHMM always requires a real control value.
function obs_distributions(hmm::ControlledEmissionHMM, ::Nothing)
    throw(
        ArgumentError(
            "ControlledEmissionHMM requires a control value; `nothing` is not valid."
        ),
    )
end

# Returns a lazy ControlledEmissions vector:
# each element is bound to `control` and responds to logdensityof(dist, obs) / rand(rng, dist)
function obs_distributions(hmm::ControlledEmissionHMM, control)
    ControlledEmissions(hmm.dists, control)
end
Base.length(hmm::ControlledEmissionHMM) = length(hmm.dists)

function Random.rand(::ControlledEmissionHMM, ::Integer)
    throw(
        ArgumentError(
            "ControlledEmissionHMM requires a control sequence; call rand(hmm, control_seq).",
        ),
    )
end

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

function StatsAPI.fit!(
    hmm::ControlledEmissionHMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector;
    control_seq=nothing,
    seq_ends::AbstractVectorOrNTuple{Int},
)
    control_seq === nothing && throw(
        ArgumentError(
            "ControlledEmissionHMM requires control_seq; call fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=...).",
        ),
    )
    return StatsAPI.fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=seq_ends)
end
