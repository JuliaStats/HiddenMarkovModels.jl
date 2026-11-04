"""
$(SIGNATURES)
"""
function initialize_forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals=true,
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hmm, control_seq[t + 1]), R)
        end
        ξ[T] = zero(trans)  # not used
    end
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    α = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, N, T)
    Bβ = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage{R,M}(γ, ξ, logL, B, α, c, β, Bβ)
end

function _forward_backward!(
    storage::ForwardBackwardStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    transition_marginals::Bool=true,
) where {R}
    (; α, β, c, γ, ξ, B, Bβ) = storage
    t1, t2 = seq_limits(seq_ends, k)

    # Forward (fill B, α, c and logL)
    _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite=true)

    # Backward
    β[:, t2] .= c[t2]
    for t in (t2 - 1):-1:t1
        Bβ[:, t + 1] .= view(B, :, t + 1) .* view(β, :, t + 1)
        βₜ = view(β, :, t)
        Bβₜ₊₁ = view(Bβ, :, t + 1)
        predict_previous_state!(βₜ, hmm, Bβₜ₊₁, control_seq[t + 1])
        lmul!(c[t], βₜ)
    end
    Bβ[:, t1] .= view(B, :, t1) .* view(β, :, t1)

    # State marginals
    γ[:, t1:t2] .= view(α, :, t1:t2) .* view(β, :, t1:t2) ./ view(c, t1:t2)'

    # Transition marginals
    if transition_marginals
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t + 1])
            mul_rows_cols!(ξ[t], view(α, :, t), trans, view(Bβ, :, t + 1))
        end
        ξ[t2] .= zero(R)
    end

    return nothing
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::ForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hmm`.

Return a tuple `(storage.γ, storage.logL)` where `storage` is of type [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hmm, obs_seq, control_seq; seq_ends, transition_marginals
    )
    forward_backward!(storage, hmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end
