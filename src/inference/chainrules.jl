_dcat(M1, M2) = cat(M1, M2; dims=3)

function _params_and_loglikelihoods(
    hmm::AbstractHMM,
    obs_seq::Vector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    init = initialization(hmm)
    trans_by_time = mapreduce(_dcat, eachindex(control_seq)[1:(end - 1)]) do t
        transition_matrix(hmm, control_seq[t + 1])
    end
    logB = mapreduce(hcat, eachindex(obs_seq, control_seq)) do t
        logdensityof.(obs_distributions(hmm, control_seq[t]), (obs_seq[t],))
    end
    return init, trans_by_time, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(logdensityof),
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    _, pullback = rrule_via_ad(
        rc, _params_and_loglikelihoods, hmm, obs_seq, control_seq; seq_ends
    )
    fb_storage = initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
    forward_backward!(fb_storage, hmm, obs_seq, control_seq; seq_ends)
    (; logL, α, γ, Bβ) = fb_storage
    N, T = length(hmm), length(obs_seq)
    R = eltype(α)

    Δinit = zeros(R, N)
    Δtrans_by_time = zeros(R, N, N, T)
    @views for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        Δinit .+= Bβ[:, t1]
        for t in t1:(t2 - 1)
            Δtrans_by_time[:, :, t] .= α[:, t] .* Bβ[:, t + 1]'
        end
    end
    ΔlogB = γ

    function logdensityof_hmm_pullback(ΔlogL)
        _, Δhmm, Δobs_seq, Δcontrol_seq = pullback((
            ΔlogL .* Δinit, ΔlogL .* Δtrans_by_time, ΔlogL .* ΔlogB
        ))
        Δlogdensityof = NoTangent()
        return Δlogdensityof, Δhmm, Δobs_seq, Δcontrol_seq
    end

    return sum(logL), logdensityof_hmm_pullback
end

# With constant controls, a single set of parameters covers every timestep.
function _params_and_loglikelihoods(
    hsmm::AbstractHSMM,
    obs_seq::Vector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int,
)
    control = control_seq[1]
    init = initialization(hsmm)
    trans = transition_matrix(hsmm, control)
    durs = duration_distributions(hsmm, control)
    D = min(max_duration, longest_sequence(seq_ends))
    durs_row = reshape(durs, 1, :)
    log_dur = duration_logdensityof.(durs_row, 1:D)
    log_surv = duration_logsurvival.(durs_row, 1:D)
    logB = mapreduce(hcat, eachindex(obs_seq, control_seq)) do t
        logdensityof.(obs_distributions(hsmm, control_seq[t]), (obs_seq[t],))
    end
    return init, trans, log_dur, log_surv, logB
end

#= Each path factors into the parameters above, so the gradient of the loglikelihood with respect
to the log of each factor is its expected number of occurrences: γ for the observations, and the
duration counts for the durations. For `init` and `trans`, it is computed without dividing by the
parameter, which may be zero. =#
function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(logdensityof),
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
)
    @argcheck uniform_controls(control_seq) "Reverse-mode differentiation of an HSMM only \
    supports constant controls."
    _, pullback = rrule_via_ad(
        rc, _params_and_loglikelihoods, hsmm, obs_seq, control_seq; seq_ends, max_duration
    )
    fb_storage = initialize_forward_backward(
        hsmm,
        obs_seq,
        control_seq;
        seq_ends,
        max_duration,
        transition_marginals=false,
        duration_counts=true,
    )
    forward_backward!(fb_storage, hsmm, obs_seq, control_seq; seq_ends)
    (; logL, γ, log_from_start, forward, completed_durations, censored_durations) =
        fb_storage
    (; log_ends) = forward
    N = length(hsmm)
    R = eltype(γ)

    Δinit = zeros(R, N)
    Δtrans = zeros(R, N, N)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        for j in 1:N
            Δinit[j] += exp(log_from_start[j, t1] - logL[k])
        end
        for t in t1:(t2 - 1), j in 1:N, i in 1:N
            i == j && continue
            Δtrans[i, j] += exp(log_ends[i, t] + log_from_start[j, t + 1] - logL[k])
        end
    end

    function logdensityof_hsmm_pullback(ΔlogL)
        _, Δhsmm, Δobs_seq, Δcontrol_seq = pullback((
            ΔlogL .* Δinit,
            ΔlogL .* Δtrans,
            ΔlogL .* completed_durations,
            ΔlogL .* censored_durations,
            ΔlogL .* γ,
        ))
        Δlogdensityof = NoTangent()
        return Δlogdensityof, Δhsmm, Δobs_seq, Δcontrol_seq
    end

    return sum(logL), logdensityof_hsmm_pullback
end
