# # Gradient Descent in HMMs

#= 
In this tutorial we explore two ways to use gradient descent when fitting HMMs:

1. Fitting parameters of an observation model that do not have closed-form updates
   (e.g., GLMs, neural networks, etc.), inside the EM algorithm.
2. Fitting the entire HMM with gradient-based optimization by leveraging automatic
   differentiation.

We will explore both approaches below.
=#

using ADTypes
using ComponentArrays
using DensityInterface
using ForwardDiff
using HiddenMarkovModels
using HMMTest #src
using LinearAlgebra
using Optim
using Random
using StableRNGs
using StatsAPI
using Test #src

rng = StableRNG(42)

#= 
For both parts of this tutorial we use a simple HMM with Gaussian observations.
Using gradient-based optimization here is overkill, but it keeps the tutorial
simple while illustrating the relevant methods.

We begin by defining a Normal observation model.
=#

mutable struct NormalModel{T}
    μ::T
    logσ::T  # unconstrained parameterization; σ = exp(logσ)
end

model_mean(mod::NormalModel) = mod.μ
stddev(mod::NormalModel) = exp(mod.logσ)

#= 
We have defined a simple probability model with two parameters: the mean and the
log of the standard deviation. Using `logσ` is intentional so we can optimize over
all real numbers without worrying about the positivity constraint on `σ`.

Next, we provide the minimal interface expected by HiddenMarkovModels.jl:
`(logdensityof, rand, fit!)`.
=#

function DensityInterface.logdensityof(mod::NormalModel, obs::T) where {T<:Real}
    s = stddev(mod)
    return - log(2π) / 2 - log(s) - ((obs - model_mean(mod)) / s)^2 / 2
end

DensityInterface.DensityKind(::NormalModel) = DensityInterface.HasDensity()

function Random.rand(rng::AbstractRNG, mod::NormalModel{T}) where {T}
    return stddev(mod) * randn(rng, T) + model_mean(mod)
end

#= 
Because we are fitting a Gaussian (and the variance can collapse to ~0), we add
weak priors to regularize the parameters. We use:
- A weak Normal prior on `μ`
- A moderate-strength Normal prior on `logσ` that pulls `σ` toward ~1
=#

const μ_prior = NormalModel(0.0, log(10.0))
const logσ_prior = NormalModel(log(1.0), log(0.5))

function neglogpost(
    μ::T,
    logσ::T,
    data::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
    μ_prior::NormalModel,
    logσ_prior::NormalModel,
) where {T<:Real}
    tmp = NormalModel(μ, logσ)

    nll = mapreduce(
        i -> -weights[i] * logdensityof(tmp, data[i]), +, eachindex(data, weights)
    )

    nll += -logdensityof(μ_prior, μ)
    nll += -logdensityof(logσ_prior, logσ)

    return nll
end

function neglogpost(
    θ::AbstractVector{T},
    data::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
    μ_prior::NormalModel,
    logσ_prior::NormalModel,
) where {T<:Real}
    μ, logσ = θ
    return neglogpost(μ, logσ, data, weights, μ_prior, logσ_prior)
end

function StatsAPI.fit!(
    mod::NormalModel, data::AbstractVector{<:Real}, weights::AbstractVector{<:Real}
)
    T = promote_type(typeof(mod.μ), typeof(mod.logσ))
    θ0 = T[T(mod.μ), T(mod.logσ)]
    obj = θ -> neglogpost(θ, data, weights, μ_prior, logσ_prior)
    result = Optim.optimize(obj, θ0, BFGS(); autodiff=AutoForwardDiff())
    mod.μ, mod.logσ = Optim.minimizer(result)
    return mod
end

#= 
Now that we have fully defined our observation model, we can create an HMM using it.
=#

init_dist = [0.2, 0.7, 0.1]
init_trans = [
    0.9 0.05 0.05;
    0.075 0.9 0.025;
    0.1 0.1 0.8
]

obs_dists = [
    NormalModel(-3.0, log(0.25)), NormalModel(0.0, log(0.5)), NormalModel(3.0, log(0.75))
]

hmm_true = HMM(init_dist, init_trans, obs_dists)

#= 
We can now generate data from this HMM.
Note: `rand(rng, hmm, T)` returns `(state_seq, obs_seq)`.
=#

state_seq, obs_seq = rand(rng, hmm_true, 10_000)

#= 
Next we fit a new HMM to this data. Baum–Welch will perform EM updates for the
HMM parameters; during the M-step, our observation model parameters are fit via
gradient-based optimization (BFGS).
=#

init_dist_guess = fill(1.0 / 3, 3)
init_trans_guess = [
    0.98 0.01 0.01;
    0.01 0.98 0.01;
    0.01 0.01 0.98
]

obs_dist_guess = [
    NormalModel(-2.0, log(1.0)), NormalModel(2.0, log(1.0)), NormalModel(0.0, log(1.0))
]

hmm_guess = HMM(init_dist_guess, init_trans_guess, obs_dist_guess)

hmm_est, lls = baum_welch(hmm_guess, obs_seq)

#= 
Great! We were able to fit the model using gradient descent inside EM.

Now we will fit the entire HMM using gradient-based optimization by leveraging
automatic differentiation. The key idea is that the forward algorithm marginalizes
out the latent states, providing the likelihood of the observations directly as a
function of all model parameters.

We can therefore optimize the negative log-likelihood returned by `forward`.
Each objective evaluation runs the forward algorithm, which can be expensive for
large datasets, but this approach allows end-to-end gradient-based fitting for
arbitrary parameterized HMMs.

To respect HMM constraints, we optimize unconstrained parameters and map them to
valid probability distributions via softmax:
- `π = softmax(ηπ)`
- each row of `A` = `softmax(row_logits)`
=#

function softmax(v::AbstractVector)
    m = maximum(v)
    ex = exp.(v .- m)
    return ex ./ sum(ex)
end

function rowsoftmax(M::AbstractMatrix)
    A = similar(M)
    for i in 1:size(M, 1)
        A[i, :] .= softmax(view(M, i, :))
    end
    return A
end

function unpack_to_hmm(θ::ComponentVector)
    K = length(θ.ηπ)

    π = softmax(θ.ηπ)
    A = rowsoftmax(θ.ηA)
    dists = [NormalModel(θ.μ[k], θ.logσ[k]) for k in 1:K]

    return HMM(π, A, dists)
end

function hmm_to_θ0(hmm::HMM)
    K = length(hmm.init)

    T = promote_type(
        eltype(hmm.init),
        eltype(hmm.trans),
        eltype(hmm.dists[1].μ),
        eltype(hmm.dists[1].logσ),
    )

    ηπ = log.(hmm.init .+ eps(T))
    ηA = log.(hmm.trans .+ eps(T))

    μ = [hmm.dists[k].μ for k in 1:K]
    logσ = [hmm.dists[k].logσ for k in 1:K]

    return ComponentVector(; ηπ=ηπ, ηA=ηA, μ=μ, logσ=logσ)
end

function negloglik_from_θ(θ::ComponentVector, obs_seq)
    hmm = unpack_to_hmm(θ)
    _, loglik = forward(hmm, obs_seq; error_if_not_finite=false)
    return -loglik[1]
end

θ0 = hmm_to_θ0(hmm_guess)
ax = getaxes(θ0)

obj(x) = negloglik_from_θ(ComponentVector(x, ax), obs_seq)

result = Optim.optimize(obj, Vector(θ0), BFGS(); autodiff=AutoForwardDiff())
hmm_est2 = unpack_to_hmm(ComponentVector(result.minimizer, ax))

#= 
We have now trained an HMM using gradient-based optimization over *all* parameters!
=#

@test isapprox(hmm_est.init, hmm_est2.init; atol=1e-3) #src
@test isapprox(hmm_est.trans, hmm_est2.trans; atol=1e-3) #src

for k in 1:length(hmm_est.init) #src
    @test isapprox(hmm_est.dists[k].μ, hmm_est2.dists[k].μ; atol=1e-3) #src
    @test isapprox(stddev(hmm_est.dists[k]), stddev(hmm_est2.dists[k]); atol=1e-3) #src
end #src
