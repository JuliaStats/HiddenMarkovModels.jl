"""
$(TYPEDEF)

The per-sequence buffers are sized from the `seq_ends` given to [`initialize_forward`](@ref), so
[`forward!`](@ref) must be called with the same `seq_ends`.

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardStorage{R}
    "filtered state marginals `α[i,t] = ℙ(X[t]=i | Y[1:t])`, segment at `t` right-censored"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    "longest sojourn duration considered (longer sojourns are truncated); each sequence uses `min(max_duration, sequence length)`"
    max_duration::Int
    # Internal buffers
    # `log_ends[j,t] = log E[j,t]`, the sojourn in `j` ends at `t`
    log_ends::Matrix{R}
    # `log_ongoing[j,t] = log F[j,t]`, the sojourn in `j` covers `t`
    log_ongoing::Matrix{R}
    # `log_prefix[t] = log ℙ(Y[1:t]) = log Σ_j F[j,t]`
    log_prefix::Vector{R}
    # `cum_log_obs[j,t] = ℓ[j,t]`, sum of the finite `log b[j,u]` for `u ≤ t`
    cum_log_obs::Matrix{R}
    # `obs_zeros[j,t] = z[j,t]`, number of `b[j,u] = 0` for `u ≤ t`
    obs_zeros::Matrix{Int}
    # `log_dur[k][d,j] = log p_j(d)` for sequence `k`
    log_dur::Vector{Matrix{R}}
    # `log_surv[k][d,j] = log S_j(d)` for sequence `k`
    log_surv::Vector{Matrix{R}}
    # `incoming[k][j] = log I[j,t]` for sequence `k` at the current `t`
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

function sequence_max_duration(max_duration::Integer, t1::Integer, t2::Integer)
    return min(max_duration, t2 - t1 + 1)
end

"""
$(SIGNATURES)
"""
function initialize_forward(
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

    #= `log_ends[j,t]`: the stay in `j` ends exactly at `t`. Only these paths can switch state
    at `t+1`, so they seed the next segments.

    `log_ongoing[j,t]`: the stay in `j` includes `t` and may last longer. These are the
    paths consistent with `Y[1:t]`, so normalizing them over `j` gives `α[:, t]`. =#
    log_ends = Matrix{R}(undef, N, T)
    log_ongoing = Matrix{R}(undef, N, T)
    log_prefix = Vector{R}(undef, T)

    # Track impossible observations separately to avoid `-Inf - (-Inf)` in prefix differences.
    cum_log_obs = Matrix{R}(undef, N, T)
    obs_zeros = Matrix{Int}(undef, N, T)

    # Per-sequence scratch space keeps parallel calls independent.
    log_dur = Vector{Matrix{R}}(undef, K)
    log_surv = Vector{Matrix{R}}(undef, K)
    incoming = Vector{Vector{R}}(undef, K)
    for k in 1:K
        t1, t2 = seq_limits(seq_ends, k)
        D = sequence_max_duration(max_duration, t1, t2)
        log_dur[k] = Matrix{R}(undef, D, N)
        log_surv[k] = Matrix{R}(undef, D, N)
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
        obs_zeros,
        log_dur,
        log_surv,
        incoming,
    )
end

#= Return the longest duration with nonzero survival for some state. Since survival is
nonincreasing, longer segments are impossible and can be skipped, e.g. for bounded supports. =#
function fill_duration_buffers!(
    log_dur::AbstractMatrix{R},
    log_surv::AbstractMatrix{R},
    hsmm::AbstractHSMM,
    control,
    max_duration::Integer,
) where {R}
    durs = duration_distributions(hsmm, control)
    support = 0
    for i in 1:length(hsmm)
        # Seed beyond the cutoff, then accumulate backward without subtracting probabilities.
        s = convert(R, duration_logsurvival(durs[i], max_duration + 1))
        for d in max_duration:-1:1
            log_dur[d, i] = duration_logdensityof(durs[i], d)
            s = logaddexp_safe(s, log_dur[d, i])
            log_surv[d, i] = s
            if d > support && !is_log_zero(s)
                support = d
            end
        end
    end
    return support
end

# A changed zero count means the segment contains an impossible observation.
@inline function segment_log_obs(
    cum_end::R, cum_start::R, zeros_end::Integer, zeros_start::Integer, log_zero::R
) where {R}
    return zeros_end == zeros_start ? cum_end - cum_start : log_zero
end

# Return false if no next state is reachable.
function accumulate_incoming!(
    incoming::AbstractVector{R},
    log_ends::AbstractMatrix{R},
    log_trans::AbstractMatrix,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    reachable = false
    for j in 1:N
        log_sum_prev = log_zero
        for i in 1:N
            #= `valid_hsmm` tolerates diagonal entries up to `eps`, and staying in `j` is not a
            new sojourn, so skip the diagonal as `joint_logdensityof` does. =#
            if i != j
                log_sum_prev = logaddexp_safe(
                    log_sum_prev, log_ends[i, t] + log_trans[i, j]
                )
            end
        end
        incoming[j] = log_sum_prev
        reachable |= !is_log_zero(log_sum_prev)
    end
    return reachable
end

#= `elementwise_log` only takes the log of stored entries, so a structural zero would read as
`log(1)`. Visit the stored entries of each column instead, which also costs O(nnz). =#
function accumulate_incoming!(
    incoming::AbstractVector{R},
    log_ends::AbstractMatrix{R},
    log_trans::SparseMatrixCSC,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    rows, vals = rowvals(log_trans), nonzeros(log_trans)
    reachable = false
    for j in 1:N
        log_sum_prev = log_zero
        for p in nzrange(log_trans, j)
            i = rows[p]
            if i != j
                log_sum_prev = logaddexp_safe(log_sum_prev, log_ends[i, t] + vals[p])
            end
        end
        incoming[j] = log_sum_prev
        reachable |= !is_log_zero(log_sum_prev)
    end
    return reachable
end

function extend_segments!(;
    log_ends::AbstractMatrix{R},
    log_ongoing::AbstractMatrix{R},
    cum_log_obs::AbstractMatrix{R},
    obs_zeros::AbstractMatrix{Int},
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
            if !is_log_zero(inc)
                log_obs = segment_log_obs(
                    cum_log_obs[j, t_end],
                    cum_log_obs[j, t],
                    obs_zeros[j, t_end],
                    obs_zeros[j, t],
                    log_zero,
                )
                if !is_log_zero(log_obs)
                    base = inc + log_obs
                    log_ends[j, t_end] = logaddexp_safe(
                        log_ends[j, t_end], base + log_dur[d, j]
                    )
                    log_ongoing[j, t_end] = logaddexp_safe(
                        log_ongoing[j, t_end], base + log_surv[d, j]
                    )
                end
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
    (; α, logL, log_ends, log_ongoing, log_prefix, cum_log_obs, obs_zeros) = storage
    log_dur = storage.log_dur[k]
    log_surv = storage.log_surv[k]
    incoming = storage.incoming[k]
    t1, t2 = seq_limits(seq_ends, k)
    # An empty sequence has likelihood one, as in the HMM forward pass.
    if t1 > t2
        logL[k] = zero(R)
        return nothing
    end
    N = length(hsmm)
    max_duration = sequence_max_duration(storage.max_duration, t1, t2)
    log_zero = convert(R, -Inf)
    uniform = uniform_controls(control_seq)

    # Keep impossible observations out of the finite prefix sums.
    for t in t1:t2
        obs_logdensities!(
            view(cum_log_obs, :, t), hsmm, obs_seq[t], control_seq[t]; error_if_not_finite
        )
        for i in 1:N
            if is_log_zero(cum_log_obs[i, t])
                cum_log_obs[i, t] = zero(R)
                obs_zeros[i, t] = 1
            else
                obs_zeros[i, t] = 0
            end
        end
    end
    for t in (t1 + 1):t2
        for i in 1:N
            cum_log_obs[i, t] += cum_log_obs[i, t - 1]
            obs_zeros[i, t] += obs_zeros[i, t - 1]
        end
    end

    log_init = log_initialization(hsmm)
    @views log_ends[:, t1:t2] .= log_zero
    @views log_ongoing[:, t1:t2] .= log_zero

    # The control at the start of a segment selects its duration distribution.
    support = fill_duration_buffers!(log_dur, log_surv, hsmm, control_seq[t1], max_duration)
    for d in 1:support
        t_end = t1 + d - 1
        for i in 1:N
            log_obs = segment_log_obs(
                cum_log_obs[i, t_end], zero(R), obs_zeros[i, t_end], 0, log_zero
            )
            if !is_log_zero(log_obs)
                base = log_init[i] + log_obs
                log_ends[i, t_end] = logaddexp_safe(
                    log_ends[i, t_end], base + log_dur[d, i]
                )
                log_ongoing[i, t_end] = logaddexp_safe(
                    log_ongoing[i, t_end], base + log_surv[d, i]
                )
            end
        end
    end

    # Reuse transitions and durations when the controls are constant.
    if uniform
        log_trans = log_transition_matrix(hsmm, control_seq[t1])
        for t in t1:(t2 - 1)
            accumulate_incoming!(incoming, log_ends, log_trans, t, N, log_zero) || continue
            extend_segments!(;
                log_ends,
                log_ongoing,
                cum_log_obs,
                obs_zeros,
                incoming,
                log_dur,
                log_surv,
                t,
                t2,
                N,
                max_duration=support,
                log_zero,
            )
        end
    else
        # Rebuilding either costs `N^2` or `N * max_duration` evaluations, so skip repeated controls.
        trans_control = control_seq[t1]
        log_trans = log_transition_matrix(hsmm, trans_control)
        filled_control = control_seq[t1]
        for t in t1:(t2 - 1)
            # Avoid building a controlled transition matrix for an unreachable segment.
            reachable = false
            for i in 1:N
                reachable |= !is_log_zero(log_ends[i, t])
            end
            reachable || continue
            if !isequal(control_seq[t + 1], trans_control)
                trans_control = control_seq[t + 1]
                log_trans = log_transition_matrix(hsmm, trans_control)
            end
            accumulate_incoming!(incoming, log_ends, log_trans, t, N, log_zero) || continue
            if !isequal(control_seq[t + 1], filled_control)
                filled_control = control_seq[t + 1]
                support = fill_duration_buffers!(
                    log_dur, log_surv, hsmm, filled_control, max_duration
                )
            end
            extend_segments!(;
                log_ends,
                log_ongoing,
                cum_log_obs,
                obs_zeros,
                incoming,
                log_dur,
                log_surv,
                t,
                t2,
                N,
                max_duration=support,
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
Uses the explicit-duration formulation of [Yu2010](@cite), Section 3.1.

Return a tuple `(storage.α, storage.logL)` where `storage` is of type
[`HSMMForwardStorage`](@ref).

`storage.α[:, t]` contains the filtered state marginals. The segment covering `t` is treated as
right-censored.

`max_duration` limits the sojourn lengths considered. It defaults to the longest sequence, which
gives the exact result but costs O(N T²) per sequence of length T. For long sequences, a smaller
value brings this down to O(N T `max_duration`), at the price of underestimating the
loglikelihood.
"""
function forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
    error_if_not_finite::Bool=true,
)
    storage = initialize_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    forward!(storage, hsmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.α, storage.logL
end
