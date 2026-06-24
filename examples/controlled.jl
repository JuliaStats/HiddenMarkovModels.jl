# # Control dependency

#=
Here, we give a example of controlled HMM (also called input-output HMM), in the special case of Markov switching regression.
=#

using DensityInterface
using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMTest  #src
using LinearAlgebra
using Random
using StableRNGs
using StatsAPI
using Test  #src

#-

rng = StableRNG(63);

# ## Model

#=
A Markov switching regression is like a classical regression, except that the weights depend on the unobserved state of an HMM.
We can represent it with the following subtype of `AbstractHMM` (see [Custom HMM structures](@ref)), which has one vector of coefficients $\beta_i$ per state.
=#

struct ControlledGaussianHMM{T} <: AbstractHMM
    init::Vector{T}
    trans::Matrix{T}
    dist_coeffs::Vector{Vector{T}}
end

#=
In state $i$ with a vector of controls $u$, our observation is given by the linear model $y \sim \mathcal{N}(\beta_i^\top u, 1)$.
Controls must be provided to both `transition_matrix` and `obs_distributions` even if they are only used by one.
=#

function HMMs.initialization(hmm::ControlledGaussianHMM)
    return hmm.init
end

function HMMs.transition_matrix(hmm::ControlledGaussianHMM, control::AbstractVector)
    return hmm.trans
end

function HMMs.obs_distributions(hmm::ControlledGaussianHMM, control::AbstractVector)
    return [Normal(dot(hmm.dist_coeffs[i], control), 1.0) for i in 1:length(hmm)]
end

#=
In this case, the transition matrix does not depend on the control.
=#

# ## Simulation

d = 3
init = [0.6, 0.4]
trans = [0.7 0.3; 0.2 0.8]
dist_coeffs = [-ones(d), ones(d)]
hmm = ControlledGaussianHMM(init, trans, dist_coeffs);

#=
Simulation requires a vector of controls, each being a vector itself with the right dimension.

Let us build several sequences of variable lengths.
=#

control_seqs = [[randn(rng, d) for t in 1:rand(100:200)] for k in 1:1000];
obs_seqs = [rand(rng, hmm, control_seq).obs_seq for control_seq in control_seqs];

obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs));

# ## Inference

#=
Not much changes from the case with simple time dependency.
=#

best_state_seq, _ = viterbi(hmm, obs_seq, control_seq; seq_ends)

# ## Learning

#=
Once more, we override the `fit!` function.
The state-related parameters are estimated in the standard way.
Meanwhile, the observation coefficients are given by the formula for [weighted least squares](https://en.wikipedia.org/wiki/Weighted_least_squares).
=#

function StatsAPI.fit!(
    hmm::ControlledGaussianHMM{T},
    fb_storage::HMMs.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends,
) where {T}
    (; γ, ξ) = fb_storage
    N = length(hmm)

    hmm.init .= 0
    hmm.trans .= 0
    for k in eachindex(seq_ends)
        t1, t2 = HMMs.seq_limits(seq_ends, k)
        hmm.init .+= γ[:, t1]
        hmm.trans .+= sum(ξ[t1:t2])
    end
    hmm.init ./= sum(hmm.init)
    for row in eachrow(hmm.trans)
        row ./= sum(row)
    end

    U = reduce(hcat, control_seq)'
    y = obs_seq
    for i in 1:N
        W = sqrt.(Diagonal(γ[i, :]))
        hmm.dist_coeffs[i] = (W * U) \ (W * y)
    end
end

#=
Now we put it to the test.
=#

init_guess = [0.5, 0.5]
trans_guess = [0.6 0.4; 0.3 0.7]
dist_coeffs_guess = [-2 * ones(d), 2 * ones(d)]
hmm_guess = ControlledGaussianHMM(init_guess, trans_guess, dist_coeffs_guess);

#-

hmm_est, loglikelihood_evolution = baum_welch(hmm_guess, obs_seq, control_seq; seq_ends)
first(loglikelihood_evolution), last(loglikelihood_evolution)

#=
How did we perform?
=#

cat(hmm_est.trans, hmm.trans; dims=3)

#-

hcat(hmm_est.dist_coeffs[1], hmm.dist_coeffs[1])

#-

hcat(hmm_est.dist_coeffs[2], hmm.dist_coeffs[2])

# ## Built-in `ControlledEmissionHMM`

#=
The pattern above defines a custom `AbstractHMM` subtype, which is the right tool when controls influence the transition matrix or the initial distribution.

When **only the emissions depend on the control**, the package ships [`ControlledEmissionHMM`](@ref) so you don't have to write the boilerplate. `init` and `trans` are stored as plain control-independent vectors and matrices, and you only have to provide a control-aware emission distribution per state.
=#

# ### Defining a control-aware emission

#=
Each emission subtypes [`ControlledEmission`](@ref) and must implement three "non-standard" three-argument methods:

- `DensityInterface.logdensityof(d, obs, control)` for inference
- `Random.rand(rng, d, control)` for sampling
- `StatsAPI.fit!(d, obs_seq, control_seq, weights)` for learning

Subtyping [`ControlledEmission`](@ref) documents the interface and provides `DensityKind` automatically. At inference time, `obs_distributions(hmm, control)` wraps each emission into a [`ControlBoundEmission`](@ref) bound to `control`, so the inner inference loops keep using the standard two-argument `logdensityof(dist, obs)` / `rand(rng, dist)` interface.

To mirror the example above, we define a Gaussian whose mean is linear in a scalar control:
=#

mutable struct LinearGaussian{T} <: ControlledEmission
    β0::T
    β1::T
    logσ::T
end

function DensityInterface.logdensityof(d::LinearGaussian, obs::Real, control::Real)
    μ = d.β0 + d.β1 * control
    σ = exp(d.logσ)
    return -log(2π) / 2 - d.logσ - ((obs - μ) / σ)^2 / 2
end

function Random.rand(rng::AbstractRNG, d::LinearGaussian, control::Real)
    μ = d.β0 + d.β1 * control
    σ = exp(d.logσ)
    return μ + σ * randn(rng)
end

#=
The `fit!` method below performs a weighted maximum-likelihood update, where the weights
are the state posteriors $\gamma_t$ supplied by Baum-Welch.
Maximizing the weighted Gaussian log-likelihood over $(\beta_0, \beta_1)$ is an ordinary
weighted least squares problem: writing $S_k = \sum_t \gamma_t u_t^k$ and
$T_k = \sum_t \gamma_t u_t^k y_t$, the normal equations have the closed-form solution used
below, with $\Delta = S_0 S_2 - S_1^2$. Given those coefficients, the variance estimate is
the weighted mean of the squared residuals, $\sigma^2 = \frac{1}{S_0} \sum_t \gamma_t (y_t - \mu_t)^2$.
=#

function StatsAPI.fit!(
    d::LinearGaussian,
    obs_seq::AbstractVector{<:Real},
    control_seq::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
)
    S0 = sum(weights)                            # S₀ = Σₜ γₜ
    S1 = sum(weights .* control_seq)             # S₁ = Σₜ γₜ·uₜ
    S2 = sum(weights .* control_seq .^ 2)        # S₂ = Σₜ γₜ·uₜ²
    T0 = sum(weights .* obs_seq)                 # T₀ = Σₜ γₜ·yₜ
    T1 = sum(weights .* control_seq .* obs_seq)  # T₁ = Σₜ γₜ·uₜ·yₜ
    Δ = S0 * S2 - S1^2                           # Δ  = S₀·S₂ - S₁²  (determinant of the normal equations)
    d.β0 = (T0 * S2 - T1 * S1) / Δ               # β₀ = (T₀·S₂ - T₁·S₁) / Δ
    d.β1 = (T1 * S0 - T0 * S1) / Δ               # β₁ = (T₁·S₀ - T₀·S₁) / Δ
    sse = sum(weights .* (obs_seq .- (d.β0 .+ d.β1 .* control_seq)) .^ 2)  # SSE = Σₜ γₜ·(yₜ - μₜ)²
    d.logσ = log(sqrt(sse / S0))                 # logσ = ½·log(SSE / S₀)
    return d
end

# ### Building the HMM

#=
Construction takes the standard parameters plus the vector of control-aware emissions — no `AbstractHMM` subtype needed.
=#

dists_lg = [LinearGaussian(-1.0, 2.0, log(0.5)), LinearGaussian(0.0, -1.0, log(1.0))]
hmm_lg = ControlledEmissionHMM(init, trans, dists_lg);

# ### Simulation

#=
A `ControlledEmissionHMM` always requires a concrete control sequence: calling `rand(hmm, T::Integer)` is not supported, since there is no sensible default control. Provide a `control_seq` of the desired length instead.
=#

control_seq_lg = randn(rng, 10000);
obs_seq_lg = rand(rng, hmm_lg, control_seq_lg).obs_seq;

# ### Inference and learning

#=
Inference works exactly as with any other `AbstractHMM`:
=#

best_state_seq_lg, _ = viterbi(hmm_lg, obs_seq_lg, control_seq_lg)

#=
For learning, `ControlledEmissionHMM` ships its own `fit!` method that re-estimates `init` and `trans` in the standard way and then calls your distribution's `fit!(d, obs_seq, control_seq, weights)` for each state — so there is nothing to override on the HMM itself.
=#

dists_lg_guess = [LinearGaussian(-0.5, 1.0, log(1.0)), LinearGaussian(0.0, 0.0, log(1.0))]
hmm_lg_guess = ControlledEmissionHMM(init_guess, trans_guess, dists_lg_guess)

hmm_lg_est, ll_lg = baum_welch(hmm_lg_guess, obs_seq_lg, control_seq_lg)
first(ll_lg), last(ll_lg)

# ## Tests  #src

@test hmm_est.dist_coeffs[1] ≈ hmm.dist_coeffs[1] atol = 0.05  #src
@test hmm_est.dist_coeffs[2] ≈ hmm.dist_coeffs[2] atol = 0.05  #src
test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)  #src
test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)  #src

@test hmm_lg.trans ≈ hmm_lg_est.trans atol = 0.10  #src
for i in 1:2  #src
    @test hmm_lg.dists[i].β0 ≈ hmm_lg_est.dists[i].β0 atol = 0.10  #src
    @test hmm_lg.dists[i].β1 ≈ hmm_lg_est.dists[i].β1 atol = 0.10  #src
    @test hmm_lg.dists[i].logσ ≈ hmm_lg_est.dists[i].logσ atol = 0.10  #src
end  #src
