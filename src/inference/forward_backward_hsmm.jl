"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`, with zero diagonal (only allocated with `transition_marginals=true`)"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    "storage of the forward pass, whose filtered marginals `forward.α` remain available"
    forward::HSMMForwardStorage{R}
    "expected number of completed sojourns in state `j` lasting `d` steps, `completed_durations[d,j]`, summed over sequences (only allocated with `duration_counts=true`)"
    completed_durations::Matrix{R}
    "expected number of right-censored final sojourns in state `j` observed for `d` steps, `censored_durations[d,j]`, summed over sequences (only allocated with `duration_counts=true`)"
    censored_durations::Matrix{R}
    # Internal buffers
    # `log_starts[j,t] = log ℙ(Y[1:t-1], a sojourn in j starts at t)`
    log_starts::Matrix{R}
    # `log_after_end[j,t] = log ℙ(Y[t+1:T] | a sojourn in j ends at t)`
    log_after_end::Matrix{R}
    # `log_from_start[j,t] = log ℙ(Y[t:T] | a sojourn in j starts at t)`
    log_from_start::Matrix{R}
    # `seg_weights[k][d]`: posterior weight of the sojourn in the current state lasting `d`
    # steps from the current start, for sequence `k`
    seg_weights::Vector{Vector{R}}
    # `seq_completed_durations[k][d,j]`: contribution of sequence `k` to `completed_durations`
    seq_completed_durations::Vector{Matrix{R}}
    # `seq_censored_durations[k][d,j]`: contribution of sequence `k` to `censored_durations`
    seq_censored_durations::Vector{Matrix{R}}
end

Base.eltype(::HSMMForwardBackwardStorage{R}) where {R} = R

# Without `duration_counts`, the counts are `0 × 0`.
function has_duration_counts(storage::HSMMForwardBackwardStorage)
    return size(storage.completed_durations, 2) > 0
end

# Without `transition_marginals`, the entries of `ξ` are left unassigned.
function has_transition_marginals(storage::HSMMForwardBackwardStorage)
    (; ξ) = storage
    return isempty(ξ) || isassigned(ξ, lastindex(ξ))
end

#= Sum `ξ[t1:t2-1]` into `ξ[t2]`, which is zero after forward-backward. Zeroing the stored entries
instead of the whole matrix keeps the sparsity pattern, so the sum does not allocate. =#
function sum_transition_marginals!(ξ::AbstractVector, t1::Integer, t2::Integer)
    t1 > t2 && return nothing  # an empty sequence has no ξ[t2] of its own
    scratch = ξ[t2]
    fill!(mynonzeros(scratch), zero(eltype(scratch)))
    for t in t1:(t2 - 1)
        scratch .+= ξ[t]
    end
    return nothing
end

"""
$(SIGNATURES)

`duration_counts=true` allocates the buffers needed by `forward_backward!(...; duration_counts=true)`.
"""
function initialize_forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=longest_sequence(seq_ends),
    transition_marginals=true,
    duration_counts=false,
)
    forward = initialize_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(forward.α)
    trans = transition_matrix(hsmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hsmm, control_seq[t + 1]), R)
        end
        # Only scratch space, but `zero` would drop the stored entries of a sparse matrix.
        ξ[T] = similar(trans, R)
        fill!(mynonzeros(ξ[T]), zero(R))
    end

    # Mass entering each state at `t`, and the likelihood of the observations after `t` given
    # that a sojourn ends at `t` or starts at `t`.
    log_starts = Matrix{R}(undef, N, T)
    log_after_end = Matrix{R}(undef, N, T)
    log_from_start = Matrix{R}(undef, N, T)

    # Posterior weights of the segments sharing a start, one entry per duration.
    seg_weights = Vector{Vector{R}}(undef, K)
    for k in 1:K
        seg_weights[k] = Vector{R}(undef, size(forward.log_dur[k], 1))
    end

    # Duration counts, accumulated per sequence for threading, and `0 × 0` unless requested.
    count_dims(D) = duration_counts ? (D, N) : (0, 0)
    seq_completed_durations = Vector{Matrix{R}}(undef, K)
    seq_censored_durations = Vector{Matrix{R}}(undef, K)
    for k in 1:K
        dims = count_dims(size(forward.log_dur[k], 1))
        seq_completed_durations[k] = zeros(R, dims)
        seq_censored_durations[k] = zeros(R, dims)
    end
    D_max = min(max_duration, longest_sequence(seq_ends))
    completed_durations = zeros(R, count_dims(D_max))
    censored_durations = zeros(R, count_dims(D_max))

    return HSMMForwardBackwardStorage{R,M}(
        γ,
        ξ,
        forward.logL,
        forward,
        completed_durations,
        censored_durations,
        log_starts,
        log_after_end,
        log_from_start,
        seg_weights,
        seq_completed_durations,
        seq_censored_durations,
    )
end

# Longest duration with nonzero survival for some state, as returned by `fill_duration_buffers!`.
function duration_support(log_surv::AbstractMatrix, max_duration::Integer)
    for d in max_duration:-1:1
        for i in axes(log_surv, 2)
            is_log_zero(log_surv[d, i]) || return d
        end
    end
    return 0
end

#=
Fill `log_from_start[:, s]` and add the posterior weights of every segment starting at `s` to `γ`.
Summing the weights over durations from the longest down gives the mass of the segments still
running at each timestep, so `γ` is built without subtracting probabilities.

With `duration_counts`, also add `w[j,s,d]` to the expected duration counts, keeping the
right-censored final segment apart since its duration is only a lower bound on the sojourn.
=#
function accumulate_segments!(
    storage::HSMMForwardBackwardStorage{R},
    k::Integer,
    s::Integer,
    t1::Integer,
    t2::Integer,
    max_duration::Integer,
    duration_counts::Bool,
) where {R}
    (; γ, log_starts, log_after_end, log_from_start, forward) = storage
    (; log_obs) = forward
    log_dur, log_surv = forward.log_dur[k], forward.log_surv[k]
    weights, logL = storage.seg_weights[k], storage.logL[k]
    completed = storage.seq_completed_durations[k]
    censored = storage.seq_censored_durations[k]
    N = size(γ, 1)
    log_zero = convert(R, -Inf)
    D = min(max_duration, t2 - s + 1)
    for j in 1:N
        log_start = log_starts[j, s]
        log_future = log_zero
        @views weights[1:D] .= zero(R)
        seg_log_obs = zero(R)
        for d in 1:D
            t_end = s + d - 1
            seg_log_obs += log_obs[j, t_end]
            # Every longer segment starting at `s` contains the same impossible observation.
            is_log_zero(seg_log_obs) && break
            # The final segment is right-censored.
            log_d = t_end == t2 ? log_surv[d, j] : log_dur[d, j]
            log_seg = seg_log_obs + log_d + log_after_end[j, t_end]
            is_log_zero(log_seg) && continue
            log_future = logaddexp_safe(log_future, log_seg)
            if !is_log_zero(log_start)
                weights[d] = exp(log_start + log_seg - logL)
            end
        end
        log_from_start[j, s] = log_future
        running = zero(R)
        for d in D:-1:1
            running += weights[d]
            γ[j, s + d - 1] += running
        end
        if duration_counts
            add_duration_counts!(completed, censored, weights, j, s, D, t2)
        end
    end
    return nothing
end

# Only the longest segment can reach `t2`, unless `max_duration` cuts it short, and it is then the
# right-censored final sojourn.
function add_duration_counts!(
    completed::AbstractMatrix,
    censored::AbstractMatrix,
    weights::AbstractVector,
    j::Integer,
    s::Integer,
    D::Integer,
    t2::Integer,
)
    is_censored = s + D - 1 == t2
    for d in 1:(is_censored ? D - 1 : D)
        completed[d, j] += weights[d]
    end
    if is_censored
        censored[D, j] += weights[D]
    end
    return nothing
end

# Fill `log_after_end[:, t]` from the segments starting at `t + 1`.
function propagate_backward!(
    log_after_end::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans,
    t::Integer,
) where {R}
    N = size(log_after_end, 1)
    for i in 1:N
        log_sum_next = convert(R, -Inf)
        for j in 1:N
            # HSMM diagonals may be close to zero rather than exactly zero.
            if i != j
                log_sum_next = logaddexp_safe(
                    log_sum_next, log_trans[i, j] + log_from_start[j, t + 1]
                )
            end
        end
        log_after_end[i, t] = log_sum_next
    end
    return nothing
end

# A structural zero would read as `log(1)`, see the sparse `accumulate_incoming!`.
function propagate_backward!(
    log_after_end::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans::SparseMatrixCSC,
    t::Integer,
) where {R}
    N = size(log_after_end, 1)
    rows, vals = rowvals(log_trans), nonzeros(log_trans)
    @views log_after_end[:, t] .= convert(R, -Inf)
    for j in 1:N
        for p in nzrange(log_trans, j)
            i = rows[p]
            if i != j
                log_after_end[i, t] = logaddexp_safe(
                    log_after_end[i, t], vals[p] + log_from_start[j, t + 1]
                )
            end
        end
    end
    return nothing
end

function transition_marginals!(
    ξₜ::AbstractMatrix{R},
    log_ends::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans,
    t::Integer,
    logL::R,
) where {R}
    N = size(ξₜ, 1)
    for j in 1:N, i in 1:N
        log_ξ = log_ends[i, t] + log_trans[i, j] + log_from_start[j, t + 1]
        ξₜ[i, j] = (i == j || is_log_zero(log_ξ)) ? zero(R) : exp(log_ξ - logL)
    end
    return nothing
end

#= Only the stored entries of `log_trans` can carry mass. `ξₜ` shares their sparsity pattern, so
writing them in place does not allocate. =#
function transition_marginals!(
    ξₜ::SparseMatrixCSC{R},
    log_ends::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans::SparseMatrixCSC,
    t::Integer,
    logL::R,
) where {R}
    N = size(ξₜ, 1)
    rows, vals = rowvals(log_trans), nonzeros(log_trans)
    fill!(nonzeros(ξₜ), zero(R))
    for j in 1:N
        for p in nzrange(log_trans, j)
            i = rows[p]
            log_ξ = log_ends[i, t] + vals[p] + log_from_start[j, t + 1]
            if i != j && !is_log_zero(log_ξ)
                ξₜ[i, j] = exp(log_ξ - logL)
            end
        end
    end
    return nothing
end

function _forward_backward!(
    storage::HSMMForwardBackwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    transition_marginals::Bool=true,
    duration_counts::Bool=false,
) where {R}
    (; γ, ξ, logL, forward, log_starts, log_after_end, log_from_start) = storage
    t1, t2 = seq_limits(seq_ends, k)
    if duration_counts
        fill!(storage.seq_completed_durations[k], zero(R))
        fill!(storage.seq_censored_durations[k], zero(R))
    end

    # Forward (fill log_obs, log_ends and logL)
    _forward!(forward, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite=true)
    t1 > t2 && return nothing

    (; log_ends) = forward
    log_dur, log_surv = forward.log_dur[k], forward.log_surv[k]
    max_duration = sequence_max_duration(forward.max_duration, t1, t2)
    uniform = uniform_controls(control_seq)
    log_init = log_initialization(hsmm)

    # Backward (fill log_starts, log_from_start, log_after_end, γ and ξ)
    @views γ[:, t1:t2] .= zero(R)
    # No observations remain after the end of the sequence.
    @views log_after_end[:, t2] .= zero(R)

    # With uniform controls, the forward pass left the duration buffers filled for `control_seq[t1]`.
    filled_control = control_seq[t1]
    support = duration_support(log_surv, max_duration)
    # Rebuilding the transitions costs `N^2` evaluations, so skip repeated controls.
    trans_control = uniform ? control_seq[t1] : control_seq[t2]
    log_trans = log_transition_matrix(hsmm, trans_control)
    for s in t2:-1:t1
        if !uniform && (s == t2 || !isequal(control_seq[s], filled_control))
            filled_control = control_seq[s]
            support = fill_duration_buffers!(
                log_dur, log_surv, hsmm, filled_control, max_duration
            )
        end
        # The control at time `s` drives the transition into time `s`.
        if !uniform && s > t1 && !isequal(control_seq[s], trans_control)
            trans_control = control_seq[s]
            log_trans = log_transition_matrix(hsmm, trans_control)
        end
        if s == t1
            copyto!(view(log_starts, :, s), log_init)
        else
            accumulate_incoming!(view(log_starts, :, s), log_ends, log_trans, s - 1)
        end
        accumulate_segments!(storage, k, s, t1, t2, support, duration_counts)
        if s > t1
            propagate_backward!(log_after_end, log_from_start, log_trans, s - 1)
            if transition_marginals
                transition_marginals!(
                    ξ[s - 1], log_ends, log_from_start, log_trans, s - 1, logL[k]
                )
            end
        end
    end
    if transition_marginals
        fill!(mynonzeros(ξ[t2]), zero(R))  # keeps the sparsity pattern
    end

    return nothing
end

# The per-sequence counts have `min(max_duration, sequence length)` rows.
function sum_duration_counts!(storage::HSMMForwardBackwardStorage{R}) where {R}
    (; completed_durations, censored_durations) = storage
    fill!(completed_durations, zero(R))
    fill!(censored_durations, zero(R))
    for (seq_completed, seq_censored) in
        zip(storage.seq_completed_durations, storage.seq_censored_durations)
        D = size(seq_completed, 1)
        @views completed_durations[1:D, :] .+= seq_completed
        @views censored_durations[1:D, :] .+= seq_censored
    end
    return nothing
end

"""
$(SIGNATURES)

With `duration_counts=true`, also fill the expected duration counts
`storage.completed_durations` and `storage.censored_durations`, from which the duration
distributions can be re-estimated. This requires the corresponding buffers, see
[`initialize_forward_backward`](@ref).
"""
function forward_backward!(
    storage::HSMMForwardBackwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
    duration_counts::Bool=false,
)
    @argcheck !duration_counts || has_duration_counts(storage)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage,
                hsmm,
                obs_seq,
                control_seq,
                seq_ends,
                k;
                transition_marginals,
                duration_counts,
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage,
                hsmm,
                obs_seq,
                control_seq,
                seq_ends,
                k;
                transition_marginals,
                duration_counts,
            )
        end
    end
    duration_counts && sum_duration_counts!(storage)
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during
sequence `obs_seq` for `hsmm`. Refer to [Yu2010](@cite) for details.

Return a tuple `(storage.γ, storage.logL)` where `storage` is of type
[`HSMMForwardBackwardStorage`](@ref).

`max_duration` limits the sojourn lengths considered, as in [`forward`](@ref).
"""
function forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, transition_marginals
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end
