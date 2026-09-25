"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])`"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    c::Vector{R}
end

"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R

const ForwardOrForwardBackwardStorage{R} = Union{
    ForwardStorage{R},ForwardBackwardStorage{R}
}

"""
$(SIGNATURES)
"""
function initialize_forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    return ForwardStorage(α, logL, B, c)
end

function _forward_digest_observation!(
    current_state_marginals::AbstractVector{<:Real},
    current_obs_likelihoods::AbstractVector{<:Real},
    hmm::AbstractHMM,
    obs,
    control;
    error_if_not_finite::Bool,
)
    a, b = current_state_marginals, current_obs_likelihoods

    obs_logdensities!(b, hmm, obs, control; error_if_not_finite)
    logm = maximum(b)
    b .= exp.(b .- logm)

    a .*= b
    c = inv(sum(a))
    lmul!(c, a)

    logL = -log(c) + logm
    return c, logL
end

function _forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool,
)
    (; α, B, c, logL) = storage
    t1, t2 = seq_limits(seq_ends, k)
    logL[k] = zero(eltype(logL))
    for t in t1:t2
        αₜ = view(α, :, t)
        Bₜ = view(B, :, t)
        if t == t1
            copyto!(αₜ, initialization(hmm))
        else
            αₜ₋₁ = view(α, :, t - 1)
            predict_next_state!(αₜ, hmm, αₜ₋₁, control_seq[t])
        end
        cₜ, logLₜ = _forward_digest_observation!(
            αₜ, Bₜ, hmm, obs_seq[t], control_seq[t]; error_if_not_finite
        )
        c[t] = cₜ
        logL[k] += logLₜ
    end

    error_if_not_finite && @argcheck isfinite(logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    error_if_not_finite::Bool=true,
)
    foreach_sequence(seq_ends) do k
        _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hmm`.

Return a tuple `(storage.α, storage.logL)` where `storage` is of type [`ForwardStorage`](@ref).
"""
function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    error_if_not_finite::Bool=true,
)
    storage = initialize_forward(hmm, obs_seq, control_seq; seq_ends)
    forward!(storage, hmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.α, storage.logL
end
