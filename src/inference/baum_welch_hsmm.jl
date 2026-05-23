function _hsmm_baum_welch_has_converged(
    logL_evolution::Vector; atol::Real, loglikelihood_increasing::Bool
)
    if length(logL_evolution) >= 2
        logL, logL_prev = logL_evolution[end], logL_evolution[end - 1]
        progress = logL - logL_prev
        if loglikelihood_increasing && progress < min(0, -atol)
            error("Loglikelihood decreased from $logL_prev to $logL in Baum-Welch")
        elseif progress < atol
            return true
        end
    end
    return false
end

"""
$(SIGNATURES)
"""
function baum_welch!(
    fb_storage::HSMMForwardBackwardStorage,
    logL_evolution::Vector,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    atol::Real,
    max_iterations::Integer,
    loglikelihood_increasing::Bool,
)
    for _ in 1:max_iterations
        forward_backward!(fb_storage, hsmm, obs_seq, control_seq; seq_ends)
        push!(logL_evolution, sum(fb_storage.logL))
        fit!(hsmm, fb_storage, obs_seq, control_seq; seq_ends)
        if _hsmm_baum_welch_has_converged(logL_evolution; atol, loglikelihood_increasing)
            break
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the Baum-Welch (EM) algorithm to estimate the parameters of an [`AbstractHSMM`](@ref) on
`obs_seq`, starting from `hsmm_guess`.

Returns `(hsmm_est, loglikelihood_evolution)`.

# Keyword arguments

- `seq_ends`: end indices of the (possibly multiple) observation sequences
- `max_duration`: maximum segment duration considered (must cover plausible runs)
- `atol`: convergence threshold on log-likelihood increase
- `max_iterations`: cap on the number of EM iterations
- `loglikelihood_increasing`: error if log-likelihood drops below `-atol`
"""
function baum_welch(
    hsmm_guess::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
    atol::Real=1e-5,
    max_iterations::Integer=100,
    loglikelihood_increasing::Bool=true,
)
    hsmm = deepcopy(hsmm_guess)
    fb_storage = initialize_hsmm_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, transition_marginals=true
    )
    logL_evolution = eltype(fb_storage.logL)[]
    sizehint!(logL_evolution, max_iterations)
    baum_welch!(
        fb_storage,
        logL_evolution,
        hsmm,
        obs_seq,
        control_seq;
        seq_ends,
        atol,
        max_iterations,
        loglikelihood_increasing,
    )
    return hsmm, logL_evolution
end
