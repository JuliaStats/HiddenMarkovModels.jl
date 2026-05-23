"""
$(TYPEDEF)

Storage for the HSMM forward algorithm.

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardStorage{R}
    "log-probability of the partial sequence with a segment ending in state `i` at time `t`: `log_α[i, t]`"
    log_α::Matrix{R}
    "one log-likelihood per observation sequence"
    logL::Vector{R}
    cum_log_obs::Matrix{R}
    "maximum sojourn duration considered for any segment"
    max_duration::Int
    dp_buffer::Vector{Matrix{R}}
end

"""
$(SIGNATURES)
"""
function initialize_hsmm_forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=50,
)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])

    log_α = fill(typemin(R), N, T)
    logL = Vector{R}(undef, K)

    # cum_log_obs[t+1, i] - cum_log_obs[t1, i] is the per-state log-likelihood of obs_seq[t1:t]
    cum_log_obs = Matrix{R}(undef, T + 1, N)

    # One dp_buffer per sequence so that sequences can be processed by parallel threads
    dp_buffer = [Matrix{R}(undef, max_duration, N) for _ in 1:K]

    return HSMMForwardStorage{R}(log_α, logL, cum_log_obs, max_duration, dp_buffer)
end

function _forward!(
    storage::HSMMForwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
) where {R}
    (; log_α, logL, cum_log_obs, max_duration) = storage
    dp_buffer = storage.dp_buffer[k]
    t1, t2 = seq_limits(seq_ends, k)
    N = length(hsmm)

    @views log_α[:, t1:t2] .= typemin(R)

    # Prefix log-likelihoods for each state across this sequence
    cum_log_obs[t1, :] .= zero(R)
    for t in t1:t2
        dists = obs_distributions(hsmm, control_seq[t])
        for i in 1:N
            val = logdensityof(dists[i], obs_seq[t])
            cum_log_obs[t + 1, i] = cum_log_obs[t, i] + val
        end
    end

    # Duration log-pmf table — assumed constant across the sequence under one control
    durs = duration_distributions(hsmm, control_seq[t1])
    for i in 1:N
        for d in 1:max_duration
            dp_buffer[d, i] = logdensityof(durs[i], d)
        end
    end

    log_init = log_initialization(hsmm)

    # Initial segments (start at t1)
    for i in 1:N
        for d in 1:min(max_duration, t2 - t1 + 1)
            t_end = t1 + d - 1
            log_obs_seg = cum_log_obs[t_end + 1, i] - cum_log_obs[t1, i]
            val = log_init[i] + dp_buffer[d, i] + log_obs_seg
            log_α[i, t_end] = logaddexp(log_α[i, t_end], val)
        end
    end

    # Internal segments: `t` is the time at which the previous segment ended
    for t in t1:(t2 - 1)
        log_trans = log_transition_matrix(hsmm, control_seq[t + 1])

        # Incoming log-prob into state j given that some other state ended at t.
        incoming_log_prob = fill(typemin(R), N)
        has_valid_path = false
        for j in 1:N
            log_sum_prev = typemin(R)
            for i in 1:N
                if i != j  # no self-transitions in HSMMs
                    log_sum_prev = logaddexp(log_sum_prev, log_α[i, t] + log_trans[i, j])
                end
            end
            incoming_log_prob[j] = log_sum_prev
            if log_sum_prev > typemin(R)
                has_valid_path = true
            end
        end

        if !has_valid_path
            continue
        end

        remaining_time = t2 - t
        for d in 1:min(max_duration, remaining_time)
            t_end = t + d
            for j in 1:N
                inc = incoming_log_prob[j]
                if inc > typemin(R)
                    log_obs_seg = cum_log_obs[t_end + 1, j] - cum_log_obs[t + 1, j]
                    val = inc + dp_buffer[d, j] + log_obs_seg
                    log_α[j, t_end] = logaddexp(log_α[j, t_end], val)
                end
            end
        end
    end

    final_logL = typemin(R)
    for i in 1:N
        final_logL = logaddexp(final_logL, log_α[i, t2])
    end
    logL[k] = final_logL

    return nothing
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::HSMMForwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to an [`AbstractHSMM`](@ref).

Returns a tuple `(storage.log_α, storage.logL)` where `storage` is of type [`HSMMForwardStorage`](@ref).

# Keyword arguments

- `seq_ends`: end indices of the (possibly multiple) observation sequences
- `max_duration`: maximum segment duration considered (must cover plausible sojourn lengths)
"""
function forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    storage = initialize_hsmm_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    forward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return storage.log_α, storage.logL
end
