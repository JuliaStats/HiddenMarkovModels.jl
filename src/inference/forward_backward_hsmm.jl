"""
$(TYPEDEF)

Storage for the HSMM forward-backward algorithm.

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct HSMMForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i, t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i, j] = ℙ(segment ends at t in state i, next segment starts in state j | Y[1:T])`"
    ξ::Vector{M}
    "expected segment-duration counts `η[d, i]`, aggregated across sequences"
    η::Matrix{R}
    "one log-likelihood per observation sequence"
    logL::Vector{R}

    # Internal log-space variables and buffers
    log_α::Matrix{R}
    log_beta::Matrix{R}
    cum_log_obs::Matrix{R}
    max_duration::Int
    dp_buffer::Vector{Matrix{R}}
    incoming_log_prob::Vector{Vector{R}}
    η_per_seq::Vector{Matrix{R}}
end

Base.eltype(::HSMMForwardBackwardStorage{R}) where {R} = R

# Accumulate state-marginal γ and segment-duration η contributions for a candidate
# segment (j, t_start, d) given its log-prefix probability. Assumes `dp_buffer` already
# holds the duration log-pmfs for segments starting at `t_start`.
@inline function _accumulate_segment!(
    γ::AbstractMatrix,
    η_local::AbstractMatrix,
    log_beta::AbstractMatrix,
    cum_log_obs::AbstractMatrix,
    dp_buffer::AbstractMatrix,
    current_logL::R,
    j::Int,
    t_start::Int,
    d::Int,
    log_prefix::R,
) where {R}
    t_end = t_start + d - 1
    log_obs_seg = cum_log_obs[t_end + 1, j] - cum_log_obs[t_start, j]
    log_seg_prob =
        log_prefix + dp_buffer[d, j] + log_obs_seg + log_beta[j, t_end] - current_logL
    prob = exp(log_seg_prob)
    if prob > zero(R)
        for τ in t_start:t_end
            γ[j, τ] += prob
        end
        η_local[d, j] += prob
    end
    return nothing
end

"""
$(SIGNATURES)
"""
function initialize_hsmm_forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=50,
    transition_marginals::Bool=true,
)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hsmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = zeros(R, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hsmm, control_seq[t + 1]), R)
        end
        ξ[T] = zero(trans)
    else
        for k in 1:K
            ξ[seq_ends[k]] = zero(trans)
        end
    end
    logL = Vector{R}(undef, K)
    η = zeros(R, max_duration, N)
    # Explicit pre-allocate-and-fill rather than comprehensions — closures over `R`
    # produce runtime dispatch that JET flags on Julia 1.10.
    η_per_seq = Vector{Matrix{R}}(undef, K)
    log_α = fill(typemin(R), N, T)
    log_beta = fill(typemin(R), N, T)
    cum_log_obs = Matrix{R}(undef, T + 1, N)
    dp_buffer = Vector{Matrix{R}}(undef, K)
    incoming_log_prob = Vector{Vector{R}}(undef, K)
    for k in 1:K
        η_per_seq[k] = zeros(R, max_duration, N)
        dp_buffer[k] = Matrix{R}(undef, max_duration, N)
        incoming_log_prob[k] = Vector{R}(undef, N)
    end

    return HSMMForwardBackwardStorage{R,M}(
        γ,
        ξ,
        η,
        logL,
        log_α,
        log_beta,
        cum_log_obs,
        max_duration,
        dp_buffer,
        incoming_log_prob,
        η_per_seq,
    )
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
    t1, t2 = seq_limits(seq_ends, k)
    N = length(hsmm)
    max_dur = storage.max_duration
    dp_buffer = storage.dp_buffer[k]
    η_local = storage.η_per_seq[k]
    fill!(η_local, zero(R))

    # Reuse the per-sequence forward to populate log_α, cum_log_obs, dp_buffer, logL
    forward_storage_view = HSMMForwardStorage{R}(
        storage.log_α,
        storage.logL,
        storage.cum_log_obs,
        storage.max_duration,
        storage.dp_buffer,
        storage.incoming_log_prob,
    )
    _forward!(forward_storage_view, hsmm, obs_seq, control_seq, seq_ends, k)

    current_logL = storage.logL[k]

    # Backward pass: log_beta[i, t] = log P(obs[t+1:t2] | a segment ENDS at t in state i)
    # Boundary at t = t2: nothing remains, so β = 1 → log β = 0.
    for i in 1:N
        storage.log_beta[i, t2] = zero(R)
    end

    for t in (t2 - 1):-1:t1
        log_trans = log_transition_matrix(hsmm, control_seq[t + 1])
        _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t + 1], max_dur, N)
        for i in 1:N
            log_sum_next = typemin(R)
            for j in 1:N
                if i != j
                    remaining = t2 - t
                    for d in 1:min(max_dur, remaining)
                        t_next_end = t + d
                        log_obs_seg =
                            storage.cum_log_obs[t_next_end + 1, j] -
                            storage.cum_log_obs[t + 1, j]
                        val =
                            log_trans[i, j] +
                            dp_buffer[d, j] +
                            log_obs_seg +
                            storage.log_beta[j, t_next_end]
                        log_sum_next = logaddexp(log_sum_next, val)
                    end
                end
            end
            storage.log_beta[i, t] = log_sum_next
        end
    end

    @views storage.γ[:, t1:t2] .= zero(R)

    log_init = log_initialization(hsmm)
    γ = storage.γ
    log_beta = storage.log_beta
    cum_log_obs2 = storage.cum_log_obs

    # A. Initial segments (start at t1) — prefix probability = π_j
    _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t1], max_dur, N)
    for j in 1:N
        for d in 1:min(max_dur, t2 - t1 + 1)
            _accumulate_segment!(
                γ,
                η_local,
                log_beta,
                cum_log_obs2,
                dp_buffer,
                current_logL,
                j,
                t1,
                d,
                log_init[j],
            )
        end
    end

    # B. Internal segments — prefix probability = Σ_i α[i, t_prev] * a_ij
    for t_start in (t1 + 1):t2
        t_prev = t_start - 1
        log_trans = log_transition_matrix(hsmm, control_seq[t_start])
        _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t_start], max_dur, N)
        for j in 1:N
            log_prefix = typemin(R)
            for i in 1:N
                if i != j
                    log_prefix = logaddexp(
                        log_prefix, storage.log_α[i, t_prev] + log_trans[i, j]
                    )
                end
            end
            if log_prefix > typemin(R)
                remaining = t2 - t_start + 1
                for d in 1:min(max_dur, remaining)
                    _accumulate_segment!(
                        γ,
                        η_local,
                        log_beta,
                        cum_log_obs2,
                        dp_buffer,
                        current_logL,
                        j,
                        t_start,
                        d,
                        log_prefix,
                    )
                end
            end
        end
    end

    if transition_marginals
        for t in t1:(t2 - 1)
            fill!(storage.ξ[t], zero(R))
            log_trans = log_transition_matrix(hsmm, control_seq[t + 1])
            _fill_dp_buffer!(dp_buffer, hsmm, control_seq[t + 1], max_dur, N)
            for i in 1:N, j in 1:N
                if i == j
                    continue
                end
                log_prefix = storage.log_α[i, t] + log_trans[i, j]
                log_suffix_sum = typemin(R)
                remaining = t2 - t
                for d in 1:min(max_dur, remaining)
                    t_next_end = t + d
                    log_obs_seg =
                        storage.cum_log_obs[t_next_end + 1, j] -
                        storage.cum_log_obs[t + 1, j]
                    val = dp_buffer[d, j] + log_obs_seg + storage.log_beta[j, t_next_end]
                    log_suffix_sum = logaddexp(log_suffix_sum, val)
                end
                storage.ξ[t][i, j] = exp(log_prefix + log_suffix_sum - current_logL)
            end
        end
        storage.ξ[t2] .= zero(R)
    end

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
    # Aggregate per-sequence segment-duration counts into the public η buffer.
    fill!(storage.η, zero(eltype(storage.η)))
    for k in eachindex(seq_ends)
        storage.η .+= storage.η_per_seq[k]
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to an [`AbstractHSMM`](@ref).

Returns a tuple `(storage.γ, storage.logL)` where `storage` is of type [`HSMMForwardBackwardStorage`](@ref).

# Keyword arguments

- `seq_ends`: end indices of the (possibly multiple) observation sequences
- `max_duration`: maximum segment duration considered (must cover plausible sojourn lengths)
- `transition_marginals`: whether to compute the transition marginals `ξ`
"""
function forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
    transition_marginals::Bool=true,
)
    storage = initialize_hsmm_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, transition_marginals
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end
