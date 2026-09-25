"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ViterbiStorage{R}
    "most likely state sequence `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "one joint loglikelihood per pair of observation sequence and most likely state sequence"
    logL::Vector{R}
    logB::Matrix{R}
    ϕ::Matrix{R}
    ψ::Matrix{Int}
end

"""
$(SIGNATURES)
"""
function initialize_viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    q = Vector{Int}(undef, T)
    logL = Vector{R}(undef, K)
    logB = Matrix{R}(undef, N, T)
    ϕ = Matrix{R}(undef, N, T)
    ψ = Matrix{Int}(undef, N, T)
    return ViterbiStorage(q, logL, logB, ϕ, ψ)
end

function _viterbi!(
    storage::ViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
) where {R}
    (; q, logB, ϕ, ψ, logL) = storage
    t1, t2 = seq_limits(seq_ends, k)

    logBₜ₁ = view(logB, :, t1)
    obs_logdensities!(logBₜ₁, hmm, obs_seq[t1], control_seq[t1])
    loginit = log_initialization(hmm)
    ϕ[:, t1] .= loginit .+ logBₜ₁

    for t in (t1 + 1):t2
        logBₜ = view(logB, :, t)
        obs_logdensities!(logBₜ, hmm, obs_seq[t], control_seq[t])
        logtrans = log_transition_matrix(hmm, control_seq[t])
        ϕₜ, ϕₜ₋₁ = view(ϕ, :, t), view(ϕ, :, t - 1)
        ψₜ = view(ψ, :, t)
        argmaxplus_transmul!(ϕₜ, ψₜ, logtrans, ϕₜ₋₁)
        ϕₜ .+= logBₜ
    end

    #= A plain loop rather than `argmax`. Inference mistakes for recursion
    when called from within the threaded reduction of `foreach_sequence`. =#
    iₘ = 1
    for i in axes(ϕ, 1)
        if ϕ[i, t2] > ϕ[iₘ, t2]
            iₘ = i
        end
    end
    q[t2] = iₘ
    logL[k] = ϕ[q[t2], t2]
    for t in (t2 - 1):-1:t1
        q[t] = ψ[q[t + 1], t + 1]
    end

    @argcheck isfinite(logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function viterbi!(
    storage::ViterbiStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
) where {R}
    foreach_sequence(seq_ends) do k
        _viterbi!(storage, hmm, obs_seq, control_seq, seq_ends, k)
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for `hmm`.

Return a tuple `(storage.q, storage.logL)` where `storage` is of type [`ViterbiStorage`](@ref).
"""
function viterbi(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    storage = initialize_viterbi(hmm, obs_seq, control_seq; seq_ends)
    viterbi!(storage, hmm, obs_seq, control_seq; seq_ends)
    return storage.q, storage.logL
end
