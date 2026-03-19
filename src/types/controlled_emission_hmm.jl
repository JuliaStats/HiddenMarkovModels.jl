"""
$(TYPEDEF)

A single emission distribution bound to a specific control value.
Wraps a control-aware distribution and a control value, exposing
the standard `logdensityof(ce, obs)` and `rand(rng, ce)` interface.

$(TYPEDFIELDS)
"""
struct ControlledEmission{D,C}
    dist::D
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

This is returned by `obs_distributions(hmm::ControlledEmissionHMM, control)` so that the
standard `obs_distributions(hmm, control)` contract is satisfied: the returned distributions
are already bound to `control` and respond to `logdensityof(dist, obs)` and `rand(rng, dist)`.

$(TYPEDFIELDS)
"""
struct ControlledEmissions{VD,C} <: AbstractVector{ControlledEmission}
    dists::VD
    control::C
end

Base.size(ce::ControlledEmissions) = size(ce.dists)
Base.getindex(ce::ControlledEmissions, i::Int) = ControlledEmission(ce.dists[i], ce.control)

"""
$(TYPEDEF)

Implementation of a Controlled HMM where control variables only influence the emission models.

$(TYPEDFIELDS)
"""
struct ControlledEmissionHMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHMM
    init::V
    trans::M
    dists::VD
    loginit::Vl
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

obs_distributions(hmm::ControlledEmissionHMM, ::Nothing) = hmm.dists
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
