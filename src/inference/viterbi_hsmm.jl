"""
$(TYPEDEF)

Storage for the HSMM Viterbi algorithm.

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMViterbiStorage{R}
    "most likely state sequence `q[t] = argmaxᵢ ℙ(X[t]=i | Y[1:T])`"
    q::Vector{Int}
    "one joint log-likelihood per observation sequence"
    logL::Vector{R}
    "best log-score of a segment ending at time `t` in state `j`: `δ[j, t]`"
    δ::Matrix{R}
    "backpointer to the previous state (0 = initial segment) for the best segment ending at `(j, t)`"
    ψ_state::Matrix{Int}
    "backpointer to the duration of the best segment ending at `(j, t)`"
    ψ_dur::Matrix{Int}
    cum_log_obs::Matrix{R}
    "maximum sojourn duration considered"
    max_duration::Int
    dp_buffer::Vector{Matrix{R}}
end

"""
$(SIGNATURES)
"""
function initialize_hsmm_viterbi(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=50,
)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    q = Vector{Int}(undef, T)
    logL = Vector{R}(undef, K)
    δ = fill(typemin(R), N, T)
    ψ_state = zeros(Int, N, T)
    ψ_dur = zeros(Int, N, T)
    cum_log_obs = Matrix{R}(undef, T + 1, N)
    dp_buffer = [Matrix{R}(undef, max_duration, N) for _ in 1:K]
    return HSMMViterbiStorage{R}(
        q, logL, δ, ψ_state, ψ_dur, cum_log_obs, max_duration, dp_buffer
    )
end

function _viterbi!(
    storage::HSMMViterbiStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
) where {R}
    (; q, logL, δ, ψ_state, ψ_dur, cum_log_obs, max_duration) = storage
    dp_buffer = storage.dp_buffer[k]
    t1, t2 = seq_limits(seq_ends, k)
    N = length(hsmm)

    @views δ[:, t1:t2] .= typemin(R)
    @views ψ_state[:, t1:t2] .= 0
    @views ψ_dur[:, t1:t2] .= 0

    # Cumulative per-state log-observation likelihoods, like in forward.
    cum_log_obs[t1, :] .= zero(R)
    for t in t1:t2
        dists = obs_distributions(hsmm, control_seq[t])
        for i in 1:N
            cum_log_obs[t + 1, i] = cum_log_obs[t, i] + logdensityof(dists[i], obs_seq[t])
        end
    end

    log_init = log_initialization(hsmm)

    # Segment DP. We split into the initial (t_start = t1) and internal (t_start > t1)
    # branches so `log_trans` has a concrete type in the internal branch.
    _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t1], max_duration, N)
    for d in 1:min(max_duration, t2 - t1 + 1)
        t_end = t1 + d - 1
        for j in 1:N
            log_obs_seg = cum_log_obs[t_end + 1, j] - cum_log_obs[t1, j]
            score = log_init[j] + dp_buffer[d, j] + log_obs_seg
            if score > δ[j, t_end]
                δ[j, t_end] = score
                ψ_state[j, t_end] = 0  # initial-segment marker
                ψ_dur[j, t_end] = d
            end
        end
    end

    for t_start in (t1 + 1):t2
        log_trans = log_transition_matrix(hsmm, control_seq[t_start])
        _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t_start], max_duration, N)
        for d in 1:min(max_duration, t2 - t_start + 1)
            t_end = t_start + d - 1
            for j in 1:N
                log_obs_seg = cum_log_obs[t_end + 1, j] - cum_log_obs[t_start, j]
                log_dur_obs = dp_buffer[d, j] + log_obs_seg
                for i in 1:N
                    if i != j
                        score = δ[i, t_start - 1] + log_trans[i, j] + log_dur_obs
                        if score > δ[j, t_end]
                            δ[j, t_end] = score
                            ψ_state[j, t_end] = i
                            ψ_dur[j, t_end] = d
                        end
                    end
                end
            end
        end
    end

    # Termination: pick best final state at t2.
    best_j = 1
    best_score = δ[1, t2]
    for j in 2:N
        if δ[j, t2] > best_score
            best_score = δ[j, t2]
            best_j = j
        end
    end
    logL[k] = best_score

    # Backtrack through (state, duration) pairs.
    j = best_j
    t_end = t2
    while t_end >= t1
        d = ψ_dur[j, t_end]
        for τ in (t_end - d + 1):t_end
            q[τ] = j
        end
        prev_state = ψ_state[j, t_end]
        if prev_state == 0
            break
        end
        t_end -= d
        j = prev_state
    end

    @argcheck isfinite(logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function viterbi!(
    storage::HSMMViterbiStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            _viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends, k)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for an [`AbstractHSMM`](@ref).

Returns a tuple `(storage.q, storage.logL)` where `storage` is of type [`HSMMViterbiStorage`](@ref).

# Keyword arguments

- `seq_ends`: end indices of the (possibly multiple) observation sequences
- `max_duration`: maximum segment duration considered
"""
function viterbi(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    storage = initialize_hsmm_viterbi(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    viterbi!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return storage.q, storage.logL
end
