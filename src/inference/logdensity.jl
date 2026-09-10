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

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hsmm`, integrating over all possible state sequences.
"""
function DensityInterface.logdensityof(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=longest_sequence(seq_ends),
)
    _, logL = forward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, error_if_not_finite=false
    )
    return sum(logL)
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

"""
$(SIGNATURES)

Compute the joint loglikelihood of `obs_seq` and `state_seq` for `hsmm`.

Each constant-state segment contributes its sojourn length to its duration distribution. The
control at the segment's first timestep determines its duration distribution, and the control at
the first timestep of the next segment determines the transition into it.

The final segment is right-censored and therefore contributes [`duration_logsurvival`](@ref)
rather than [`duration_logdensityof`](@ref).
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
        # Initialization
        loginit = log_initialization(hsmm)
        logL += loginit[state_seq[t1]]
        # Observations
        for t in t1:t2
            dists = obs_distributions(hsmm, control_seq[t])
            logL += logdensityof(dists[state_seq[t]], obs_seq[t])
        end
        # Durations and transitions, segment by segment
        for (t_start, t_end) in StateSegments(state_seq, t1, t2)
            i = state_seq[t_start]
            d = t_end - t_start + 1
            durations = duration_distributions(hsmm, control_seq[t_start])
            if t_end == t2
                # Last segment of the sequence: right-censored, no outgoing transition.
                logL += duration_logsurvival(durations[i], d)
            else
                logL += duration_logdensityof(durations[i], d)
                logtrans = log_transition_matrix(hsmm, control_seq[t_end + 1])
                logL += logtrans[i, state_seq[t_end + 1]]
            end
        end
    end
    return logL
end
