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
