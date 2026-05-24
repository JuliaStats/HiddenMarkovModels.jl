# # Hidden Semi-Markov Models

#=
A Hidden Semi-Markov Model (HSMM) extends an HMM by replacing the implicit geometric
sojourn distribution of a standard HMM with an explicit per-state distribution over
sojourn durations. This is useful when very short (or very long) state visits are
implausible, e.g., in segmentation, behavioral modeling, or rainfall regimes.

The package exposes HSMMs through types and functions that mirror the HMM API. The
core inference functions ([`forward`](@ref), [`viterbi`](@ref),
[`forward_backward`](@ref), [`baum_welch`](@ref)) dispatch on
[`AbstractHSMM`](@ref) to a segment-based dynamic program that accounts for sojourn
times.
=#

using Distributions
using HiddenMarkovModels
using HMMTest  #src
using LinearAlgebra
using Random
using StableRNGs
using Statistics
using Test  #src

#-

rng = StableRNG(63);

# ## Model

#=
An [`HSMM`](@ref) has the three usual HMM ingredients plus one more:
- a vector `init` of initial state probabilities,
- a matrix `trans` of state transition probabilities (the diagonal is zeroed at
  construction; sojourn lengths are governed by `durations`, not by repeated
  self-transitions),
- a vector `dists` of observation distributions, one per state,
- a vector `durations` of sojourn distributions, one per state.

The duration types live in [`AbstractDurationDistribution`](@ref). The package
ships three concrete duration distributions in core: [`GeometricDuration`](@ref),
[`PoissonDuration`](@ref), and [`NegBinomialDuration`](@ref).
=#

init = [0.4, 0.3, 0.3]
trans = [0.0 0.6 0.4; 0.5 0.0 0.5; 0.5 0.5 0.0]
dists = [Normal(-3.0, 0.7), Normal(0.0, 0.7), Normal(3.0, 0.7)]
durations = [PoissonDuration(4.0), PoissonDuration(4.0), PoissonDuration(4.0)]
hsmm = HSMM(init, trans, dists, durations)

# ## Simulation

#=
[`rand`](@ref) takes either a sequence length `T` or a control sequence and returns
a named tuple `(; state_seq, obs_seq, duration_seq)`. The extra `duration_seq` field
records, for each timestep, the duration that was drawn for the sojourn containing
that timestep (so consecutive timesteps in the same sojourn share the same value).
=#

T = 30
sim = rand(rng, hsmm, T)
sim.state_seq'

#-

sim.duration_seq'

#=
Because self-transitions are forbidden, each state visit lasts for the
corresponding sojourn length, then the chain transitions to a different state.
=#

# ## Inference

#=
The [`viterbi`](@ref) algorithm returns the most likely state sequence and its
joint log-likelihood. The HSMM version performs the maximization over both the
state at each segment boundary and the segment duration.
=#

best_state_seq, best_joint_loglikelihood = viterbi(hsmm, sim.obs_seq);
only(best_joint_loglikelihood)

#=
With well-separated observation distributions and modest noise, decoding usually
performs well.
=#

mean(best_state_seq .== sim.state_seq)

#=
The [`forward`](@ref) algorithm returns the matrix `log_α[j, t] =` log-probability
that a segment ends at time `t` in state `j`, alongside the log-likelihood of the
observation sequence.
=#

log_α, obs_loglikelihood_f = forward(hsmm, sim.obs_seq);
only(obs_loglikelihood_f)

#=
[`forward_backward`](@ref) returns the matrix of smoothed state marginals
`γ[i, t] = ℙ(X_t = i | Y_{1:T})` (which sums to one across states at every
timestep) and the log-likelihood — same value as the forward pass.
=#

γ, obs_loglikelihood_fb = forward_backward(hsmm, sim.obs_seq);
only(obs_loglikelihood_fb)

#-

all(isapprox.(sum(γ; dims=1), 1; atol=1e-6))

#=
[`logdensityof`](@ref) is the usual thin wrapper around the forward pass for
observation-sequence log-likelihoods.
=#

logdensityof(hsmm, sim.obs_seq)

# ## The `max_duration` keyword

#=
HSMM inference is parameterized by a `max_duration` keyword (default `50`) that
caps the longest segment considered. Increasing it makes inference slower (the
inner DP is O(N²·T·max_duration)) but admits longer sojourns; decreasing it can
clip rare long runs and bias the likelihood. Pick a value comfortably above the
plausible runs in your data.
=#

log_α_short, _ = forward(hsmm, sim.obs_seq; max_duration=20);
log_α_long, _ = forward(hsmm, sim.obs_seq; max_duration=80);
size(log_α_short) == size(log_α_long)

# ## Learning

#=
[`baum_welch`](@ref) on an HSMM re-fits the initialization, transition matrix,
observation distributions, **and** duration distributions in one pass per EM
iteration. As with the HMM case it needs a reasonable starting guess.
=#

init_guess = [0.1, 0.5, 0.4]
trans_guess = [0.0 0.3 0.7; 0.7 0.0 0.3; 0.3 0.7 0.0]
dists_guess = [Normal(-2.0, 1.0), Normal(0.5, 1.0), Normal(2.0, 1.0)]
durations_guess = [PoissonDuration(2.0), PoissonDuration(6.0), PoissonDuration(8.0)]
hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, durations_guess);

#=
Fit on a longer realization.
=#

long_sim = rand(rng, hsmm, 1500)
hsmm_est, loglikelihood_evolution = baum_welch(hsmm_guess, long_sim.obs_seq);

#=
The log-likelihood is monotone non-decreasing across iterations.
=#

first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
Estimated observation means are close to the truth.
=#

map(d -> d.μ, hcat(obs_distributions(hsmm_est), obs_distributions(hsmm)))

#=
And so are the duration distributions.
=#

map(d -> d.λ, hcat(duration_distributions(hsmm_est), duration_distributions(hsmm)))

# ## Multiple sequences

#=
HSMMs use the same concatenation convention as HMMs: pass a flattened
`obs_seq` plus a `seq_ends` keyword giving the cumulative end indices of each
sequence. [`seq_limits`](@ref) recovers the slice for a given sequence.
=#

nb_seqs = 100
obs_seqs = [rand(rng, hsmm, rand(rng, 100:200)).obs_seq for _ in 1:nb_seqs]
obs_seq_concat = reduce(vcat, obs_seqs)
seq_ends = cumsum(length.(obs_seqs))
first(seq_ends, 5)'

#-

hsmm_est_concat, _ = baum_welch(hsmm_guess, obs_seq_concat; seq_ends);
map(d -> d.μ, hcat(obs_distributions(hsmm_est_concat), obs_distributions(hsmm)))

#=
Permutation note: like HMMs, HSMMs are not identifiable up to a relabeling of
their states. State `i` in the estimated model need not correspond to state `i`
in the true model.
=#

# ## Choosing a duration distribution

#=
- [`GeometricDuration`](@ref) — implicit sojourn distribution of a standard HMM;
  this should give the same results as an HMM. Included for benchmarking and as a baseline when 
  comparing different duration distributions. 
- [`PoissonDuration`](@ref) — concentrated around a mode, good for state visits
  that have a typical length but some variability.
- [`NegBinomialDuration`](@ref) — overdispersed (variance > mean); useful when
  some sojourns are much longer than typical.

All three implement `DensityInterface.logdensityof`, `Random.rand`, and `StatsAPI.fit!`,
so they work without `Distributions.jl`. To define your own duration distribution,
subtype [`AbstractDurationDistribution`](@ref) and implement those three methods.
=#

# ## Tests  #src

@test startswith(string(hsmm), "Hidden")  #src
@test length(sim.state_seq) == T  #src
@test length(sim.duration_seq) == T  #src
@test all(isapprox.(sum(γ; dims=1), 1; atol=1e-6))  #src
@test all(diff(loglikelihood_evolution) .>= -1e-8)  #src
@test loglikelihood_evolution[end] > loglikelihood_evolution[1]  #src

control_seq_tests = fill(nothing, last(seq_ends))  #src
test_coherent_algorithms(rng, hsmm, control_seq_tests; seq_ends, hsmm_guess)  #src
test_type_stability(rng, hsmm, control_seq_tests; seq_ends, hsmm_guess)  #src
test_allocations(rng, hsmm, control_seq_tests; seq_ends, hsmm_guess)  #src
