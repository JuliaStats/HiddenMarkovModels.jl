# Mirror of `_params_and_loglikelihoods` for HSMMs: extracts the three "local" parameter
# blocks the segment-DP forward consumes — initial probabilities, per-timestep transition
# matrices, per-(state, t) observation log-densities — and lets `rrule_via_ad` propagate
# cotangents back to the HSMM's init/trans/dists parameters.
#
# !!! note "Duration parameters are not differentiated by this rrule"
#     Duration distributions are intentionally **not** included in this helper's outputs,
#     so the cotangent for `hsmm.durations` returned through this rrule is empty. The
#     reason is a Zygote bug: pulling back through `[MutableStruct(λ)][1].field` (which
#     is what `duration_distributions(hsmm)[i]` reduces to under our `mutable struct`
#     duration types) double-counts the gradient. Our duration types are mutable so that
#     `StatsAPI.fit!` can update them in place during Baum–Welch — the same property
#     that breaks Zygote's pullback. Use `ForwardDiff.gradient` to get correct gradients
#     of `logdensityof` w.r.t. duration parameters; ForwardDiff has no such limitation.
function _hsmm_params_and_logvals(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    init = initialization(hsmm)
    trans_by_time = mapreduce(_dcat, eachindex(control_seq)[1:(end - 1)]) do t
        transition_matrix(hsmm, control_seq[t + 1])
    end
    logB = mapreduce(hcat, eachindex(obs_seq, control_seq)) do t
        logdensityof.(obs_distributions(hsmm, control_seq[t]), (obs_seq[t],))
    end
    return init, trans_by_time, logB
end

function ChainRulesCore.rrule(
    rc::RuleConfig,
    ::typeof(logdensityof),
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    _, pullback = rrule_via_ad(
        rc, _hsmm_params_and_logvals, hsmm, obs_seq, control_seq; seq_ends
    )

    fb_storage = initialize_hsmm_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, transition_marginals=true
    )
    forward_backward!(fb_storage, hsmm, obs_seq, control_seq; seq_ends)
    (; γ, ξ, logL) = fb_storage
    N, T = length(hsmm), length(obs_seq)
    R = eltype(γ)

    init = initialization(hsmm)
    Δinit = zeros(R, N)
    Δtrans_by_time = zeros(R, N, N, T - 1)

    for k in eachindex(seq_ends)
        t1, _ = seq_limits(seq_ends, k)
        # ∂ log P(Y) / ∂ π_i = γ[i, t1] / π_i. Sum across sequences (per-sequence γ at the
        # respective start). Skip entries where init_i = 0 so structural zeros don't
        # produce NaN gradients.
        for i in 1:N
            if init[i] > zero(R)
                Δinit[i] += γ[i, t1] / init[i]
            end
        end
    end

    # ∂ log P(Y) / ∂ a_ij(t→t+1) = ξ[t][i,j] / a_ij(t→t+1). The diagonal (no
    # self-transitions) and any structural zeros in `trans` make this 0/0 — guarded so
    # the cotangent there is zero.
    for t in 1:(T - 1)
        trans_t = transition_matrix(hsmm, control_seq[t + 1])
        ξt = ξ[t]
        for i in 1:N, j in 1:N
            a = trans_t[i, j]
            if a > zero(a)
                Δtrans_by_time[i, j, t] = ξt[i, j] / a
            end
        end
    end

    # ∂ log P(Y) / ∂ log b_i(o_t) = γ[i, t]
    ΔlogB = γ

    function logdensityof_hsmm_pullback(ΔlogL)
        _, Δhsmm, Δobs_seq, Δcontrol_seq = pullback((
            ΔlogL .* Δinit, ΔlogL .* Δtrans_by_time, ΔlogL .* ΔlogB
        ))
        Δlogdensityof = NoTangent()
        return Δlogdensityof, Δhsmm, Δobs_seq, Δcontrol_seq
    end

    return sum(logL), logdensityof_hsmm_pullback
end
