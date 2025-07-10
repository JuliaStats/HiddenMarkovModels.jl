# # Interfaces

#=
Here we discuss how to extend the observation distributions or model fitting to satisfy specific needs.
=#

using DensityInterface
using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMTest  #src
using LinearAlgebra
using Random: Random, AbstractRNG
using StableRNGs
using StatsAPI
using Test  #src

#-

rng = StableRNG(63);

# ## Custom distributions

#=
In an `HMM` object, the observation distributions do not need to come from Distributions.jl.
They only need to implement three methods:
- `Random.rand(rng, dist)` for sampling
- `DensityInterface.logdensityof(dist, obs)` for inference
- `StatsAPI.fit!(dist, obs_seq, weight_seq)` for learning

When learning, the one-state marginal $ \gamma_{i,t} = \mathbb{P}(X_t=i | Y_{1:T}) $ will be passed to `StatsAPI.fit!` through the `weight_seq` argument.
This is analogous to the weights vector accepted by `Distributions.fit_mle`; see [juliastats.org/Distributions.jl/stable/fit/#Maximum-Likelihood-Estimation](https://juliastats.org/Distributions.jl/stable/fit/#Distributions.fit_mle-Tuple%7BAny,%20Any,%20Any%7D)

In addition, the observations can be arbitrary Julia types.
So let's construct a distribution that generates stuff.
=#

struct Stuff{T}
    quantity::T
end

#=
The associated distribution will only be a wrapper for a normal distribution on the quantity.
=#

mutable struct StuffDist{T}
    quantity_mean::T
end

#=
Simulation is fairly easy.
=#

function Random.rand(rng::AbstractRNG, dist::StuffDist)
    quantity = dist.quantity_mean + randn(rng)
    return Stuff(quantity)
end

#=
It is important to declare to [DensityInterface.jl](https://github.com/JuliaMath/DensityInterface.jl) that the custom distribution has a density, thanks to the following trait.
The logdensity itself can be computed up to an additive constant without issue.
=#

DensityInterface.DensityKind(::StuffDist) = HasDensity()

function DensityInterface.logdensityof(dist::StuffDist, obs::Stuff)
    return -abs2(obs.quantity - dist.quantity_mean) / 2
end

#=
Finally, the fitting procedure must happen in place, and take a sequence of weighted samples.
=#

function StatsAPI.fit!(
    dist::StuffDist, obs_seq::AbstractVector{<:Stuff}, weight_seq::AbstractVector{<:Real}
)
    dist.quantity_mean =
        sum(weight_seq[k] * obs_seq[k].quantity for k in eachindex(obs_seq, weight_seq)) /
        sum(weight_seq)
    return nothing
end

#=
Let's put it to the test.
=#

init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dists = [StuffDist(-1.0), StuffDist(+1.0)]
hmm = HMM(init, trans, dists);

#=
When we sample an observation sequence, we get a vector of `Stuff`.
=#

state_seq, obs_seq = rand(rng, hmm, 100)
eltype(obs_seq)

#=
And we can pass these observations to all of our inference algorithms.
=#

viterbi(hmm, obs_seq)

#=
If we implement `fit!`, Baum-Welch also works seamlessly.
=#

init_guess = [0.5, 0.5]
trans_guess = [0.6 0.4; 0.3 0.7]
dists_guess = [StuffDist(-1.1), StuffDist(+1.1)]
hmm_guess = HMM(init_guess, trans_guess, dists_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm, obs_seq)
first(loglikelihood_evolution), last(loglikelihood_evolution)

#-

obs_distributions(hmm_est)

#-

transition_matrix(hmm_est)

#=
If you want more sophisticated examples, check out [`HiddenMarkovModels.LightDiagNormal`](@ref) and [`HiddenMarkovModels.LightCategorical`](@ref), which are designed to be fast and allocation-free.
=#

# ## Tests  #src

seq_ends = cumsum(rand(rng, 100:200, 100));  #src
control_seq = fill(nothing, last(seq_ends));  #src
test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)  #src
test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)  #src
test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)  #src

# ## Custom HMM structures

#=
In some scenarios, the vanilla Baum-Welch algorithm is not exactly what we want.
For instance, we might have a prior on the parameters of our model, which we want to apply during the fitting step of the iterative procedure.
Then we need to create a new type that satisfies the `AbstractHMM` interface.

Let's make a simpler version of the built-in `HMM`, with a prior saying that each transition has already been observed a certain number of times.
Such a prior can be very useful to regularize estimation and avoid numerical instabilities.
It amounts to drawing every row of the transition matrix from a Dirichlet distribution, where each Dirichlet parameter is one plus the number of times the corresponding transition has been observed.
=#

struct PriorHMM{T,D} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dists::Vector{D}
    trans_prior_count::Int
end

#=
The basic requirements for `AbstractHMM` are the following three functions: [`initialization`](@ref), [`transition_matrix`](@ref) and [`obs_distributions`](@ref).
=#

HiddenMarkovModels.initialization(hmm::PriorHMM) = hmm.init
HiddenMarkovModels.transition_matrix(hmm::PriorHMM) = hmm.trans
HiddenMarkovModels.obs_distributions(hmm::PriorHMM) = hmm.dists

#=
It is also possible to override [`logdensityof(hmm)`](@ref) and specify a prior loglikelihood for the model itself.
If we forget to implement this, the loglikelihood computed in Baum-Welch will be missing a term, and thus it might decrease.
=#

function DensityInterface.logdensityof(hmm::PriorHMM)
    prior = Dirichlet(fill(hmm.trans_prior_count + 1, length(hmm)))
    return sum(logdensityof(prior, row) for row in eachrow(transition_matrix(hmm)))
end

#=
Finally, we must redefine the specific method of [`fit!`](@ref) that is used during Baum-Welch re-estimation.
This function takes as inputs:

- the `hmm` itself
- a `fb_storage` of type [`HiddenMarkovModels.ForwardBackwardStorage`](@ref) containing the results of the forward-backward algorithm.
- the same inputs as `baum_welch` for multiple sequences

The goal is to modify `hmm` in-place, updating parameters with their maximum likelihood estimates given current inference results.
We will make use of the fields `fb_storage.γ` and `fb_storage.ξ`, which contain the state and transition marginals `γ[i, t]` and `ξ[t][i, j]` at each time step.
=#

function StatsAPI.fit!(
    hmm::PriorHMM,
    fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends,
)
    ## initialize to defaults without observations
    hmm.init .= 0
    hmm.trans .= hmm.trans_prior_count  # our prior comes into play, otherwise 0
    ## iterate over observation sequences
    for k in eachindex(seq_ends)
        ## get sequence endpoints
        t1, t2 = seq_limits(seq_ends, k)
        ## add estimated number of initializations in each state 
        hmm.init .+= fb_storage.γ[:, t1]
        ## add estimated number of transitions between each pair of states
        hmm.trans .+= sum(fb_storage.ξ[t1:t2])
    end
    ## normalize
    hmm.init ./= sum(hmm.init)
    hmm.trans ./= sum(hmm.trans; dims=2)

    for i in 1:length(hmm)
        ## weigh each sample by the marginal probability of being in state i
        weight_seq = fb_storage.γ[i, :]
        ## fit observation distribution i using those weights
        fit!(hmm.dists[i], obs_seq, weight_seq)
    end

    ## perform a few checks on the model
    @assert HMMs.valid_hmm(hmm)
    return nothing
end

#=
!!! warning "When distributions don't comply"
    Note that some distributions, such as those from Distributions.jl:
    - do not support in-place fitting
    - expect different input formats, e.g. matrices instead of a vector of vectors
    The function [`HiddenMarkovModels.fit_in_sequence!`](@ref) is a replacement for `fit!`,  designed to handle Distributions.jl without committing type piracy.
    Check out its source code, and overload it for your other distributions too if they do not support in-place fitting.
=#

#=
Now let's see that everything works, even with our custom distribution from before.
=#

trans_prior_count = 10
prior_hmm_guess = PriorHMM(init_guess, trans_guess, dists_guess, trans_prior_count);

#-

prior_hmm_est, prior_logl_evolution = baum_welch(prior_hmm_guess, obs_seq)
first(prior_logl_evolution), last(prior_logl_evolution)

#=
As we can see, the transition matrix for our Bayesian version is slightly more spread out, although this effect would nearly disappear with enough data.
=#

cat(transition_matrix(hmm_est), transition_matrix(prior_hmm_est); dims=3)

#- 

std(vec(transition_matrix(hmm_est))) < std(vec(transition_matrix(hmm)))

# ## Tests  #src

@test std(vec(transition_matrix(hmm_est))) < std(vec(transition_matrix(hmm)))  #src
