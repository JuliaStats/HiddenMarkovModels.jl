function valid_prob_vec(p::AbstractVector{T}) where {T}
    return minimum(p) >= zero(T) && sum(p) ≈ one(T)
end

function valid_trans_mat(A::AbstractMatrix)
    return size(A, 1) == size(A, 2) && all(valid_prob_vec, eachrow(A))
end

function valid_dists(d::AbstractVector)
    for i in eachindex(d)
        if DensityKind(d[i]) == NoDensity()
            return false
        end
    end
    return true
end

"""
    valid_hmm(hmm)

Perform some checks to rule out obvious inconsistencies with an `AbstractHMM` object.

Dispatches on [`AbstractLatentStateModel`](@ref) so the same structural checks (probability
initialization, stochastic transition matrix, matching dimensions, density-bearing observation
distributions) back both [`valid_hmm`](@ref) and [`valid_hsmm`](@ref).
"""
function valid_hmm(hmm::AbstractLatentStateModel, control=nothing)
    init = initialization(hmm)
    trans = transition_matrix(hmm, control)
    dists = obs_distributions(hmm, control)
    if !(length(init) == length(dists) == size(trans, 1) == size(trans, 2))
        return false
    elseif !valid_prob_vec(init)
        return false
    elseif !valid_trans_mat(trans)
        return false
    elseif !valid_dists(dists)
        return false
    end
    return true
end

"""
    valid_hsmm(hsmm, control=nothing)

Perform some checks to rule out obvious inconsistencies with an `AbstractHSMM` object.

On top of the [`valid_hmm`](@ref)-style checks (probability initialization, stochastic transition
matrix, matching dimensions, density-bearing observation distributions), this verifies:

- the transition matrix has a zero diagonal (no self-transitions);
- there is one density-bearing duration distribution per state.
"""
function valid_hsmm(hsmm::AbstractHSMM, control=nothing)
    valid_hmm(hsmm, control) || return false

    trans = transition_matrix(hsmm, control)
    durations = duration_distributions(hsmm, control)
    N = length(hsmm)

    # One duration distribution per state.
    length(durations) == N || return false

    # No self-transitions (allowing for numerical noise).
    all(<=(eps(eltype(trans))), Diagonal(trans)) || return false

    # Duration distributions must carry a density.
    return valid_dists(durations)
end
