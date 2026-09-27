"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`, with zero diagonal"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    "storage of the forward pass, whose filtered marginals `forward.α` remain available"
    forward::HSMMForwardStorage{R}
    log_starts::Matrix{R}
    log_after_end::Matrix{R}
    log_from_start::Matrix{R}
    seg_weights::Vector{Vector{R}}
end

"""
$(SIGNATURES)
"""
function initialize_forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=longest_sequence(seq_ends),
    transition_marginals=true,
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
        ξ[T] = zero(trans)  # not used
    end

    # Mass entering each state at `t`, and the likelihood of the observations after `t` given
    # that a sojourn ends at `t` or starts at `t`.
    log_starts = Matrix{R}(undef, N, T)
    log_after_end = Matrix{R}(undef, N, T)
    log_from_start = Matrix{R}(undef, N, T)

    # Posterior weights of the segments sharing a start, one entry per duration.
    seg_weights = [Vector{R}(undef, size(forward.log_dur[k], 1)) for k in 1:K]

    return HSMMForwardBackwardStorage{R,M}(
        γ, ξ, forward.logL, forward, log_starts, log_after_end, log_from_start, seg_weights
    )
end

#=
Fill `log_from_start[:, s]` and add the posterior weights of every segment starting at `s` to `γ`.
Summing the weights over durations from the longest down gives the mass of the segments still
running at each timestep, so `γ` is built without subtracting probabilities.
=#
function accumulate_segments!(;
    γ::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    weights::AbstractVector{R},
    log_starts::AbstractMatrix{R},
    log_after_end::AbstractMatrix{R},
    cum_log_obs::AbstractMatrix{R},
    obs_zeros::AbstractMatrix{Int},
    log_dur::AbstractMatrix{R},
    log_surv::AbstractMatrix{R},
    s::Integer,
    t1::Integer,
    t2::Integer,
    N::Integer,
    max_duration::Integer,
    logL::R,
    log_zero::R,
) where {R}
    D = min(max_duration, t2 - s + 1)
    for j in 1:N
        cum_start = s == t1 ? zero(R) : cum_log_obs[j, s - 1]
        zeros_start = s == t1 ? 0 : obs_zeros[j, s - 1]
        log_start = log_starts[j, s]
        log_future = log_zero
        for d in 1:D
            t_end = s + d - 1
            weights[d] = zero(R)
            log_obs = segment_log_obs(
                cum_log_obs[j, t_end], cum_start, obs_zeros[j, t_end], zeros_start, log_zero
            )
            is_log_zero(log_obs) && continue
            # The final segment is right-censored.
            log_d = t_end == t2 ? log_surv[d, j] : log_dur[d, j]
            log_seg = log_obs + log_d + log_after_end[j, t_end]
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
    end
    return nothing
end

# Fill `log_after_end[:, t]` from the segments starting at `t + 1`.
function propagate_backward!(
    log_after_end::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans,
    t::Integer,
    N::Integer,
    log_zero::R,
) where {R}
    for i in 1:N
        log_sum_next = log_zero
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

function transition_marginals!(
    ξₜ::AbstractMatrix{R},
    log_ends::AbstractMatrix{R},
    log_from_start::AbstractMatrix{R},
    log_trans,
    t::Integer,
    N::Integer,
    logL::R,
) where {R}
    for j in 1:N, i in 1:N
        log_ξ = log_ends[i, t] + log_trans[i, j] + log_from_start[j, t + 1]
        ξₜ[i, j] = (i == j || is_log_zero(log_ξ)) ? zero(R) : exp(log_ξ - logL)
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
) where {R}
    (; γ, ξ, logL, forward, log_starts, log_after_end, log_from_start) = storage
    t1, t2 = seq_limits(seq_ends, k)

    # Forward (fill log_ends, cum_log_obs, obs_zeros and logL)
    _forward!(forward, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite=true)
    t1 > t2 && return nothing

    (; log_ends, cum_log_obs, obs_zeros) = forward
    log_dur = forward.log_dur[k]
    log_surv = forward.log_surv[k]
    weights = storage.seg_weights[k]
    N = length(hsmm)
    max_duration = sequence_max_duration(forward.max_duration, t1, t2)
    log_zero = convert(R, -Inf)
    uniform = uniform_controls(control_seq)
    log_init = log_initialization(hsmm)

    @views γ[:, t1:t2] .= zero(R)
    # No observations remain after the end of the sequence.
    @views log_after_end[:, t2] .= zero(R)

    # With uniform controls, the forward pass left the duration buffers filled for `control_seq[t1]`.
    filled_control = control_seq[t1]
    for s in t2:-1:t1
        if !uniform && (s == t2 || !isequal(control_seq[s], filled_control))
            filled_control = control_seq[s]
            fill_duration_buffers!(log_dur, log_surv, hsmm, filled_control, max_duration, N)
        end
        # The control at time `s` drives the transition into time `s`.
        log_trans = log_transition_matrix(hsmm, uniform ? control_seq[t1] : control_seq[s])
        if s == t1
            copyto!(view(log_starts, :, s), log_init)
        else
            accumulate_incoming!(
                view(log_starts, :, s), log_ends, log_trans, s - 1, N, log_zero
            )
        end
        accumulate_segments!(;
            γ,
            log_from_start,
            weights,
            log_starts,
            log_after_end,
            cum_log_obs,
            obs_zeros,
            log_dur,
            log_surv,
            s,
            t1,
            t2,
            N,
            max_duration,
            logL=logL[k],
            log_zero,
        )
        if s > t1
            propagate_backward!(
                log_after_end, log_from_start, log_trans, s - 1, N, log_zero
            )
            if transition_marginals
                transition_marginals!(
                    ξ[s - 1], log_ends, log_from_start, log_trans, s - 1, N, logL[k]
                )
            end
        end
    end
    transition_marginals && (ξ[t2] .= zero(R))

    return nothing
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::HSMMForwardBackwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hsmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hsmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    end
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
