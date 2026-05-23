"""
$(SIGNATURES)

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hmm`, integrating over all possible state sequences.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    _, logL = forward(hmm, obs_seq, control_seq; seq_ends, error_if_not_finite=false)
    return sum(logL)
end

"""
$(SIGNATURES)

Run the segment-based forward algorithm to compute the loglikelihood of `obs_seq` for `hsmm`, integrating over all possible state and sojourn sequences.
"""
function DensityInterface.logdensityof(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    _, logL = forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    return sum(logL)
end

"""
$(SIGNATURES)

Compute the joint log-likelihood ``\\log \\mathbb{P}(X_{1:T}, Y_{1:T})`` of an observation
sequence `obs_seq` together with a state sequence `state_seq` under `hsmm`.

`state_seq` is decomposed into sojourns (maximal runs of identical states); the joint
log-likelihood is the sum of:

- the initial state log-probability ``\\log \\pi_{s_1}``,
- the log-probabilities of state transitions between consecutive sojourns,
- the duration log-pmfs for each sojourn under the corresponding
  [`duration_distributions`](@ref) at the sojourn start,
- the per-timestep observation log-densities ``\\log b_{s_t}(y_t)``.

!!! note
    The final sojourn is treated as having ended exactly at the sequence boundary; no
    right-censoring correction is applied to its duration term. This matches the
    convention used by [`forward`](@ref) on HSMMs.
"""
function joint_logdensityof(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    logL = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        # Initial state
        init = initialization(hsmm)
        logL += log(init[state_seq[t1]])

        # Walk the sojourn structure. `sojourn_start` is the index where the current
        # sojourn began; whenever the state changes (or we hit the sequence end) we
        # close it out and add its duration log-pmf, plus the transition into the
        # next sojourn when there is one.
        sojourn_start = t1
        for t in (t1 + 1):t2
            if state_seq[t] != state_seq[sojourn_start]
                sojourn_state = state_seq[sojourn_start]
                duration = t - sojourn_start
                dur_dists = duration_distributions(hsmm, control_seq[sojourn_start])
                logL += logdensityof(dur_dists[sojourn_state], duration)
                log_trans = log_transition_matrix(hsmm, control_seq[t])
                logL += log_trans[sojourn_state, state_seq[t]]
                sojourn_start = t
            end
        end
        # Close out the final sojourn (covers t2).
        sojourn_state = state_seq[sojourn_start]
        duration = t2 - sojourn_start + 1
        dur_dists = duration_distributions(hsmm, control_seq[sojourn_start])
        logL += logdensityof(dur_dists[sojourn_state], duration)

        # Per-timestep observation log-densities.
        for t in t1:t2
            obs_dists = obs_distributions(hsmm, control_seq[t])
            logL += logdensityof(obs_dists[state_seq[t]], obs_seq[t])
        end
    end
    return logL
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the the joint loglikelihood of `obs_seq` and `state_seq` for `hmm`.
"""
function joint_logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Initialization
        init = initialization(hmm)
        logL += log(init[state_seq[t1]])
        # Transitions
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t + 1])
            logL += log(trans[state_seq[t], state_seq[t + 1]])
        end
        # Observations
        for t in t1:t2
            dists = obs_distributions(hmm, control_seq[t])
            logL += logdensityof(dists[state_seq[t]], obs_seq[t])
        end
    end
    return logL
end
