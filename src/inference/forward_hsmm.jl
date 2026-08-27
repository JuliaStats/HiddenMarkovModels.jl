"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])`"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    "longest sojourn duration considered for any segment"
    max_duration::Int
    log_ends::Matrix{R}
    log_ongoing::Matrix{R}
    log_prefix::Vector{R}
    cum_log_obs::Matrix{R}
    log_dur::Vector{Matrix{R}}
    log_surv::Vector{Matrix{R}}
    incoming::Vector{Vector{R}}
end

function longest_sequence(seq_ends::AbstractVectorOrNTuple{Int})
    L = 0
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        L = max(L, t2 - t1 + 1)
    end
    return L
end

function uniform_controls(control_seq::AbstractVector)
    return control_seq isa AbstractFill || eltype(control_seq) === Nothing
end

"""
$(SIGNATURES)
"""
function initialize_hsmm_forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=longest_sequence(seq_ends),
)
    @argcheck max_duration >= 1
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])

    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)

    # Completed segments drive the recursion; ongoing segments give the filtered marginals.
    log_ends = Matrix{R}(undef, N, T)
    log_ongoing = Matrix{R}(undef, N, T)
    log_prefix = Vector{R}(undef, T)

    # Observation log-density prefix sums are kept separate for each sequence.
    cum_log_obs = Matrix{R}(undef, N, T)

    # Per-sequence scratch space keeps parallel calls independent.
    log_dur = Vector{Matrix{R}}(undef, K)
    log_surv = Vector{Matrix{R}}(undef, K)
    incoming = Vector{Vector{R}}(undef, K)
    for k in 1:K
        log_dur[k] = Matrix{R}(undef, max_duration, N)
        log_surv[k] = Matrix{R}(undef, max_duration, N)
        incoming[k] = Vector{R}(undef, N)
    end

    return HSMMForwardStorage{R}(
        α,
        logL,
        max_duration,
        log_ends,
        log_ongoing,
        log_prefix,
        cum_log_obs,
        log_dur,
        log_surv,
        incoming,
    )
end

function fill_duration_buffers!(
    log_dur::AbstractMatrix{R},
    log_surv::AbstractMatrix{R},
    hsmm::AbstractHSMM,
    control,
    max_duration::Integer,
    N::Integer,
) where {R}
    durs = duration_distributions(hsmm, control)
    for i in 1:N
        log_head = convert(R, -Inf)
        for d in 1:max_duration
            # Complementing the head keeps survival probabilities independent of max_duration.
            log_surv[d, i] = log1mexp(min(log_head, zero(R)))
            log_dur[d, i] = duration_logdensityof(durs[i], d)
            log_head = logaddexp(log_head, log_dur[d, i])
        end
    end
    return nothing
end

# Return false if no next state is reachable.
function accumulate_incoming!(
    incoming::AbstractVector{R},
    log_ends::AbstractMatrix{R},
    log_trans,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    reachable = false
    for j in 1:N
        log_sum_prev = log_zero
        for i in 1:N
            # HSMM diagonals may be close to zero rather than exactly zero.
            if i != j
                log_sum_prev = logaddexp(log_sum_prev, log_ends[i, t] + log_trans[i, j])
            end
        end
        incoming[j] = log_sum_prev
        reachable |= log_sum_prev > log_zero
    end
    return reachable
end

function extend_segments!(
    log_ends::AbstractMatrix{R},
    log_ongoing::AbstractMatrix{R},
    cum_log_obs::AbstractMatrix{R},
    incoming::AbstractVector{R},
    log_dur::AbstractMatrix{R},
    log_surv::AbstractMatrix{R},
    t::Integer,
    t2::Integer,
    N::Integer,
    max_duration::Integer,
    log_zero::R,
) where {R}
    for d in 1:min(max_duration, t2 - t)
        t_end = t + d
        for j in 1:N
            inc = incoming[j]
            if inc > log_zero
                # The prefix difference covers observations t+1:t_end.
                base = inc + cum_log_obs[j, t_end] - cum_log_obs[j, t]
                log_ends[j, t_end] = logaddexp(log_ends[j, t_end], base + log_dur[d, j])
                log_ongoing[j, t_end] = logaddexp(
                    log_ongoing[j, t_end], base + log_surv[d, j]
                )
            end
        end
    end
    return nothing
end

function normalize_marginals!(
    α::AbstractMatrix{R},
    log_prefix::AbstractVector{R},
    log_ongoing::AbstractMatrix{R},
    t1::Integer,
    t2::Integer,
    N::Integer,
) where {R}
    for t in t1:t2
        logm = log_ongoing[1, t]
        for i in 2:N
            logm = max(logm, log_ongoing[i, t])
        end
        if isfinite(logm)
            s = zero(R)
            for i in 1:N
                s += exp(log_ongoing[i, t] - logm)
            end
            log_prefix[t] = logm + log(s)
            for i in 1:N
                α[i, t] = exp(log_ongoing[i, t] - log_prefix[t])
            end
        else
            # Avoid NaNs when no state can produce this prefix.
            log_prefix[t] = logm
            for i in 1:N
                α[i, t] = zero(R)
            end
        end
    end
    return nothing
end

function _forward!(
    storage::HSMMForwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool,
) where {R}
    (; α, logL, log_ends, log_ongoing, log_prefix, cum_log_obs, max_duration) = storage
    log_dur = storage.log_dur[k]
    log_surv = storage.log_surv[k]
    incoming = storage.incoming[k]
    t1, t2 = seq_limits(seq_ends, k)
    N = length(hsmm)
    log_zero = convert(R, -Inf)
    uniform = uniform_controls(control_seq)

    # Cumulative observation log densities.
    for t in t1:t2
        obs_logdensities!(
            view(cum_log_obs, :, t), hsmm, obs_seq[t], control_seq[t]; error_if_not_finite
        )
    end
    for t in (t1 + 1):t2
        for i in 1:N
            cum_log_obs[i, t] += cum_log_obs[i, t - 1]
        end
    end

    log_init = log_initialization(hsmm)
    @views log_ends[:, t1:t2] .= log_zero
    @views log_ongoing[:, t1:t2] .= log_zero

    # The control at the start of a segment selects its duration distribution.
    fill_duration_buffers!(log_dur, log_surv, hsmm, control_seq[t1], max_duration, N)
    for d in 1:min(max_duration, t2 - t1 + 1)
        t_end = t1 + d - 1
        for i in 1:N
            base = log_init[i] + cum_log_obs[i, t_end]
            log_ends[i, t_end] = logaddexp(log_ends[i, t_end], base + log_dur[d, i])
            log_ongoing[i, t_end] = logaddexp(log_ongoing[i, t_end], base + log_surv[d, i])
        end
    end

    # Reuse transitions and durations when the controls are constant.
    if uniform
        log_trans = log_transition_matrix(hsmm, control_seq[t1])
        for t in t1:(t2 - 1)
            accumulate_incoming!(incoming, log_ends, log_trans, t, N, log_zero) || continue
            extend_segments!(
                log_ends,
                log_ongoing,
                cum_log_obs,
                incoming,
                log_dur,
                log_surv,
                t,
                t2,
                N,
                max_duration,
                log_zero,
            )
        end
    else
        for t in t1:(t2 - 1)
            # Avoid building a controlled transition matrix for an unreachable segment.
            reachable = false
            for i in 1:N
                reachable |= log_ends[i, t] > log_zero
            end
            reachable || continue
            log_trans = log_transition_matrix(hsmm, control_seq[t + 1])
            accumulate_incoming!(incoming, log_ends, log_trans, t, N, log_zero) || continue
            fill_duration_buffers!(
                log_dur, log_surv, hsmm, control_seq[t + 1], max_duration, N
            )
            extend_segments!(
                log_ends,
                log_ongoing,
                cum_log_obs,
                incoming,
                log_dur,
                log_surv,
                t,
                t2,
                N,
                max_duration,
                log_zero,
            )
        end
    end

    normalize_marginals!(α, log_prefix, log_ongoing, t1, t2, N)
    logL[k] = log_prefix[t2]
    error_if_not_finite && @argcheck isfinite(logL[k])
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
    error_if_not_finite::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hsmm`.

Return a tuple `(storage.α, storage.logL)` where `storage` is of type
[`HSMMForwardStorage`](@ref).

`storage.α[:, t]` contains the filtered state marginals. The segment covering `t` is treated as
right-censored.

`max_duration` limits the sojourn lengths considered. It defaults to the longest sequence, which
gives the exact result; smaller values trade accuracy for speed.
"""
function forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
    error_if_not_finite::Bool=true,
)
    storage = initialize_hsmm_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    forward!(storage, hsmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.α, storage.logL
end
