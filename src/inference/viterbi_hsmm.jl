"""
$(TYPEDEF)

The per-sequence buffers are sized from the `seq_ends` given to [`initialize_viterbi`](@ref), so
[`viterbi!`](@ref) must be called with the same `seq_ends`.

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMViterbiStorage{R}
    "most likely state sequence `q = argmax ℙ(X[1:T] | Y[1:T])`"
    q::Vector{Int}
    "one joint loglikelihood per pair of observation sequence and most likely state sequence"
    logL::Vector{R}
    "longest sojourn duration considered (longer sojourns are truncated); each sequence uses `min(max_duration, sequence length)`"
    max_duration::Int
    # Internal buffers
    # `log_best_ends[j,t] = log E*[j,t]`, best path whose sojourn in `j` ends at `t`
    log_best_ends::Matrix{R}
    # `best_durations[j,t]`: duration of the sojourn in `j` ending at `t` on that path
    best_durations::Matrix{Int}
    # `best_previous[j,s]`: state before the sojourn in `j` starting at `s` on the best path
    best_previous::Matrix{Int}
    # `cum_log_obs[j,t] = ℓ[j,t]`, sum of the finite `log b[j,u]` for `u ≤ t`
    cum_log_obs::Matrix{R}
    # `obs_zeros[j,t] = z[j,t]`, number of `b[j,u] = 0` for `u ≤ t`
    obs_zeros::Matrix{Int}
    # `log_dur[k][d,j] = log p_j(d)` for sequence `k`
    log_dur::Vector{Matrix{R}}
    # `log_surv[k][d,j] = log S_j(d)` for sequence `k`
    log_surv::Vector{Matrix{R}}
    # `incoming[k][j] = log I*[j,t]` for sequence `k` at the current `t`
    incoming::Vector{Vector{R}}
end

"""
$(SIGNATURES)
"""
function initialize_viterbi(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=longest_sequence(seq_ends),
)
    @argcheck max_duration >= 1
    N, T = length(hsmm), length(obs_seq)
    R = eltype(hsmm, obs_seq[1], control_seq[1])

    q = Vector{Int}(undef, T)
    logL = Vector{R}(undef, length(seq_ends))
    log_best_ends = Matrix{R}(undef, N, T)
    best_durations = Matrix{Int}(undef, N, T)
    best_previous = Matrix{Int}(undef, N, T)
    cum_log_obs = Matrix{R}(undef, N, T)
    obs_zeros = Matrix{Int}(undef, N, T)
    log_dur, log_surv, incoming = sequence_buffers(R, N, seq_ends, max_duration)

    return HSMMViterbiStorage{R}(
        q,
        logL,
        max_duration,
        log_best_ends,
        best_durations,
        best_previous,
        cum_log_obs,
        obs_zeros,
        log_dur,
        log_surv,
        incoming,
    )
end

# Like `accumulate_incoming!` with a max instead of a sum. Return false if no state is reachable.
function best_incoming!(
    incoming::AbstractVector{R},
    best_previous::AbstractMatrix{Int},
    log_best_ends::AbstractMatrix{R},
    log_trans::AbstractMatrix,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    reachable = false
    for j in 1:N
        log_max_prev, i_max = log_zero, 0
        for i in 1:N
            # HSMM diagonals may be close to zero rather than exactly zero.
            if i != j
                log_prev = log_best_ends[i, t] + log_trans[i, j]
                if log_prev > log_max_prev
                    log_max_prev, i_max = log_prev, i
                end
            end
        end
        incoming[j] = log_max_prev
        best_previous[j, t + 1] = i_max
        reachable |= !is_log_zero(log_max_prev)
    end
    return reachable
end

# A structural zero would read as `log(1)`, see the sparse `accumulate_incoming!`.
function best_incoming!(
    incoming::AbstractVector{R},
    best_previous::AbstractMatrix{Int},
    log_best_ends::AbstractMatrix{R},
    log_trans::SparseMatrixCSC,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    rows, vals = rowvals(log_trans), nonzeros(log_trans)
    reachable = false
    for j in 1:N
        log_max_prev, i_max = log_zero, 0
        for p in nzrange(log_trans, j)
            i = rows[p]
            if i != j
                log_prev = log_best_ends[i, t] + vals[p]
                if log_prev > log_max_prev
                    log_max_prev, i_max = log_prev, i
                end
            end
        end
        incoming[j] = log_max_prev
        best_previous[j, t + 1] = i_max
        reachable |= !is_log_zero(log_max_prev)
    end
    return reachable
end

# Like `extend_segments!` with a max instead of a sum, for the segments starting at `s`.
function extend_best_segments!(;
    log_best_ends::AbstractMatrix{R},
    best_durations::AbstractMatrix{Int},
    cum_log_obs::AbstractMatrix{R},
    obs_zeros::AbstractMatrix{Int},
    incoming::AbstractVector{R},
    log_dur::AbstractMatrix{R},
    log_surv::AbstractMatrix{R},
    s::Integer,
    t1::Integer,
    t2::Integer,
    N::Integer,
    max_duration::Integer,
    log_zero::R,
) where {R}
    for j in 1:N
        inc = incoming[j]
        is_log_zero(inc) && continue
        cum_start = s == t1 ? zero(R) : cum_log_obs[j, s - 1]
        zeros_start = s == t1 ? 0 : obs_zeros[j, s - 1]
        for d in 1:min(max_duration, t2 - s + 1)
            t_end = s + d - 1
            log_obs = segment_log_obs(
                cum_log_obs[j, t_end], cum_start, obs_zeros[j, t_end], zeros_start, log_zero
            )
            # Later segments contain the same impossible observation.
            is_log_zero(log_obs) && break
            # The final segment is right-censored.
            log_d = t_end == t2 ? log_surv[d, j] : log_dur[d, j]
            log_path = inc + log_obs + log_d
            if log_path > log_best_ends[j, t_end]
                log_best_ends[j, t_end] = log_path
                best_durations[j, t_end] = d
            end
        end
    end
    return nothing
end

function _viterbi!(
    storage::HSMMViterbiStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
) where {R}
    (; q, logL, log_best_ends, best_durations, best_previous, cum_log_obs, obs_zeros) =
        storage
    log_dur = storage.log_dur[k]
    log_surv = storage.log_surv[k]
    incoming = storage.incoming[k]
    t1, t2 = seq_limits(seq_ends, k)
    # An empty sequence has likelihood one, as in the HSMM forward pass.
    if t1 > t2
        logL[k] = zero(R)
        return nothing
    end
    N = length(hsmm)
    max_duration = sequence_max_duration(storage.max_duration, t1, t2)
    log_zero = convert(R, -Inf)
    uniform = uniform_controls(control_seq)

    cumulate_log_obs!(
        cum_log_obs,
        obs_zeros,
        hsmm,
        obs_seq,
        control_seq,
        t1,
        t2,
        N;
        error_if_not_finite=true,
    )
    @views log_best_ends[:, t1:t2] .= log_zero

    # Forward sweep over the segment starts, as in the forward pass.
    filled_control = control_seq[t1]
    support = fill_duration_buffers!(log_dur, log_surv, hsmm, filled_control, max_duration)
    # Rebuilding the transitions costs `N^2` evaluations, so skip repeated controls.
    trans_control = control_seq[t1]
    log_trans = log_transition_matrix(hsmm, trans_control)
    copyto!(incoming, log_initialization(hsmm))
    for s in t1:t2
        if s > t1
            # Avoid building a controlled transition matrix for an unreachable segment.
            reachable = false
            for i in 1:N
                reachable |= !is_log_zero(log_best_ends[i, s - 1])
            end
            reachable || continue
            # The control at time `s` drives the transition into time `s`.
            if !uniform && !isequal(control_seq[s], trans_control)
                trans_control = control_seq[s]
                log_trans = log_transition_matrix(hsmm, trans_control)
            end
            best_incoming!(
                incoming, best_previous, log_best_ends, log_trans, s - 1, N, log_zero
            ) || continue
            if !uniform && !isequal(control_seq[s], filled_control)
                filled_control = control_seq[s]
                support = fill_duration_buffers!(
                    log_dur, log_surv, hsmm, filled_control, max_duration
                )
            end
        end
        extend_best_segments!(;
            log_best_ends,
            best_durations,
            cum_log_obs,
            obs_zeros,
            incoming,
            log_dur,
            log_surv,
            s,
            t1,
            t2,
            N,
            max_duration=support,
            log_zero,
        )
    end

    # Backtrack segment by segment, from the best right-censored final segment.
    j = argmax(view(log_best_ends, :, t2))
    logL[k] = log_best_ends[j, t2]
    @argcheck isfinite(logL[k])
    t_end = t2
    while true
        s = t_end - best_durations[j, t_end] + 1
        for t in s:t_end
            q[t] = j
        end
        s == t1 && break
        j, t_end = best_previous[j, s], s - 1
    end
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

Apply the Viterbi algorithm to infer the most likely state sequence corresponding to `obs_seq` for
`hsmm`. The final segment of each sequence is right-censored, as in
[`joint_logdensityof`](@ref).

Return a tuple `(storage.q, storage.logL)` where `storage` is of type
[`HSMMViterbiStorage`](@ref).

`max_duration` limits the sojourn lengths considered, as in [`forward`](@ref).
"""
function viterbi(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
)
    storage = initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    viterbi!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return storage.q, storage.logL
end
