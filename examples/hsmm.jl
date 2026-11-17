# # Semi-Markov models

#=
Here we show how to use hidden semi-Markov models (HSMMs), whose states last for an explicit random duration.
=#

using Distributions
using HiddenMarkovModels
using Random
using StableRNGs
using Test  #src

#-

rng = StableRNG(63);

# ## Model

#=
In an HMM, the time spent in a state is implicit: at each step, the chain stays with probability $a_{i,i}$, so sojourns follow a geometric distribution.
This is often a poor fit, for instance when a state has a typical length, or when short stays are rare.

An HSMM makes the sojourn durations explicit.
The [`HSMM`](@ref) type has the same attributes as an [`HMM`](@ref), plus a vector `dur_dists` of duration distributions, one for each state:
- the transition matrix `trans` must have a zero diagonal, since staying in a state is handled by its duration
- each element of `dur_dists` is a discrete distribution on $\{0, 1, 2, \dots\}$, which models the sojourn length minus one, because every sojourn lasts at least one step
=#

init = [0.6, 0.3, 0.1]
trans = [0.0 0.7 0.3; 0.5 0.0 0.5; 0.8 0.2 0.0]
dists = [Normal(-2.0), Normal(0.0), Normal(2.0)]
dur_dists = [Poisson(4.0), Poisson(9.0), Poisson(1.0)]
hsmm = HSMM(init, trans, dists, dur_dists)

#=
With these distributions, the mean sojourn lengths are 5, 10 and 2 steps.
=#

mean.(dur_dists) .+ 1

# ## Simulation

#=
Simulation works as for an HMM, except that [`rand`](@ref) also returns the sampled sojourn lengths.
=#

T = 30
state_seq, obs_seq, duration_seq = rand(rng, hsmm, T);

#=
The state sequence is made of runs, whose lengths follow the duration distributions.
=#

state_seq'

#=
At each step, `duration_seq` gives the length of the current sojourn.
The last one can be longer than what remains of the sequence, since the sequence cuts it short.
=#

duration_seq'

# ## Inference

#=
Every inference algorithm from [Basics](@ref) applies to HSMMs, with the same outputs.
The most likely state sequence:
=#

best_state_seq, best_joint_loglikelihood = viterbi(hsmm, obs_seq);
(state_seq .== best_state_seq)'

#=
The smoothed state marginals and the loglikelihood:
=#

smoothed_state_marginals, obs_seq_loglikelihood = forward_backward(hsmm, obs_seq);
only(obs_seq_loglikelihood)

#=
And the loglikelihood alone:
=#

logdensityof(hsmm, obs_seq)

#=
These algorithms consider every possible start and length for each sojourn, so their cost grows like $T^2$ for a sequence of length $T$.
The keyword argument `max_duration` caps the sojourn lengths that are considered, which brings the cost down to $T \times$ `max_duration`.
It should be well above the typical sojourn length, since longer sojourns are dropped and the loglikelihood is then underestimated.
=#

logdensityof(hsmm, obs_seq; max_duration=25)

#-

logdensityof(hsmm, obs_seq; max_duration=3)

#=
See [Formulas](@ref) for the details of the algorithms.
=#

# ## Learning

#=
The Baum-Welch algorithm ([`baum_welch`](@ref)) also estimates the duration distributions.
As for HMMs, it needs a starting point close enough to the true model, and benefits from several sequences.
=#

init_guess = [0.4, 0.3, 0.3]
trans_guess = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0]
dists_guess = [Normal(-1.5), Normal(0.5), Normal(1.5)]
dur_dists_guess = [Poisson(2.0), Poisson(2.0), Poisson(2.0)]
hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, dur_dists_guess);

#-

obs_seqs = [rand(rng, hsmm, 100).obs_seq for _ in 1:200]
seq_ends = cumsum(length.(obs_seqs))
hsmm_est, loglikelihood_evolution = baum_welch(
    hsmm_guess, reduce(vcat, obs_seqs); seq_ends, max_duration=40
);

#=
The loglikelihood still increases at every iteration.
=#

first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
The duration distributions are recovered, along with the other parameters.
=#

hcat(mean.(duration_distributions(hsmm_est)), mean.(dur_dists))

#-

cat(transition_matrix(hsmm_est), transition_matrix(hsmm); dims=3)

#-

hcat(mean.(obs_distributions(hsmm_est)), mean.(obs_distributions(hsmm)))

#=
The last sojourn of each sequence is cut short by the end of the sequence.
Baum-Welch accounts for it, so the durations are not biased downwards, even with short sequences.
=#

# ## Duration distributions

#=
Any discrete distribution on $\{0, 1, 2, \dots\}$ from Distributions.jl can be used for inference.
For learning, the duration distributions are fitted with weights, which Distributions.jl supports for some families only, such as `Geometric`, `Poisson` or `Binomial`.
Others, such as `NegativeBinomial`, need a custom method for [`HiddenMarkovModels.fit_duration_in_sequence!`](@ref).

With `Geometric` durations, an HSMM is equivalent to an HMM, so you only need an HSMM when the durations are not geometric.
=#

# ## Tests  #src

@test length.(values(rand(hsmm, T))) == (T, T, T)  #src
@test all(>=(0), diff(loglikelihood_evolution))  #src
@test logdensityof(hsmm, obs_seq; max_duration=3) < logdensityof(hsmm, obs_seq)  #src
@test all(  #src
    isapprox.(mean.(duration_distributions(hsmm_est)), mean.(dur_dists); rtol=0.1)  #src
)  #src
@test all(isapprox.(transition_matrix(hsmm_est), transition_matrix(hsmm); atol=0.1))  #src
