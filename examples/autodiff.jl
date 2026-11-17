# # Differentiation

#=
Here we show how to compute gradients of the observation sequence loglikelihood with respect to various inputs.
=#

using ComponentArrays
using DensityInterface
using Distributions
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HMMTest  #src
using LinearAlgebra
using Random: Random, AbstractRNG
using StableRNGs
using StatsAPI
using Test  #src
using Zygote: Zygote

#-

rng = StableRNG(63);

# ## Diffusion HMM

#=
To play around with automatic differentiation, we define a simple controlled HMM.
=#

struct DiffusionHMM{V1<:AbstractVector,M2<:AbstractMatrix,V3<:AbstractVector} <: AbstractHMM
    init::V1
    trans::M2
    means::V3
end

#=
Both its transition matrix and its vector of observation means result from a convex combination between the corresponding field and a base value (aka diffusion).
The coefficient $\lambda$ of this convex combination is given as a control.
=#

HMMs.initialization(hmm::DiffusionHMM) = hmm.init

function HMMs.transition_matrix(hmm::DiffusionHMM, λ::Number)
    N = length(hmm)
    return (1 - λ) * hmm.trans + λ * ones(N, N) / N
end

function HMMs.obs_distributions(hmm::DiffusionHMM, λ::Number)
    return [Normal((1 - λ) * hmm.means[i] + λ * 0) for i in 1:length(hmm)]
end

#=
We now construct an instance of this object and draw samples from it.
=#

init = [0.6, 0.4]
trans = [0.7 0.3; 0.3 0.7]
means = [-1.0, 1.0]
hmm = DiffusionHMM(init, trans, means);

#=
It is essential that the controls are taken between $0$ and $1$.
=#

control_seqs = [rand(rng, 3), rand(rng, 5)];
obs_seqs = [rand(rng, hmm, control_seqs[k]).obs_seq for k in 1:2];

control_seq = reduce(vcat, control_seqs)
obs_seq = reduce(vcat, obs_seqs)
seq_ends = cumsum(length.(obs_seqs));

# ## What to differentiate?

#=
The key function we are interested in is the loglikelihood of the observation sequence.
We can differentiate it with respect to
- the model itself (`hmm`), or more precisely its parameters
- the observation sequence (`obs_seq`)
- the control sequence (`control_seq`).
- but not with respect to the sequence limits (`seq_ends`), which are discrete.
=#

logdensityof(hmm, obs_seq, control_seq; seq_ends)

#=
To ensure compatibility with backends that only accept a single input, we wrap all parameters inside a `ComponentVector` from [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl), and define a new function to differentiate.
=#

parameters = ComponentVector(; init, trans, means)

function f(parameters::ComponentVector, obs_seq, control_seq; seq_ends)
    new_hmm = DiffusionHMM(parameters.init, parameters.trans, parameters.means)
    return logdensityof(new_hmm, obs_seq, control_seq; seq_ends)
end;

f(parameters, obs_seq, control_seq; seq_ends)

# ## Forward mode

#=
Since all of our code is type-generic, it is amenable to forward-mode automatic differentiation with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

Because ForwardDiff.jl only accepts a single input, we must compute derivatives one at a time.
=#

∇parameters_forwarddiff = ForwardDiff.gradient(
    _parameters -> f(_parameters, obs_seq, control_seq; seq_ends), parameters
)

#-

∇obs_forwarddiff = ForwardDiff.gradient(
    _obs_seq -> f(parameters, _obs_seq, control_seq; seq_ends), obs_seq
)

#-

∇control_forwarddiff = ForwardDiff.gradient(
    _control_seq -> f(parameters, obs_seq, _control_seq; seq_ends), control_seq
)

#=
These values will serve as ground truth when we compare with reverse mode.
=#

# ## Reverse mode with Zygote.jl

#=
In the presence of many parameters, reverse mode automatic differentiation of the loglikelihood will be much more efficient.
The package includes a handwritten chain rule for `logdensityof`, which means backends like [Zygote.jl](https://github.com/FluxML/Zygote.jl) can be used out of the box.
The rule also covers HSMMs, as long as the controls are constant.
Using it, we can compute all derivatives at once.
=#

∇all_zygote = Zygote.gradient(
    (_a, _b, _c) -> f(_a, _b, _c; seq_ends), parameters, obs_seq, control_seq
);

∇parameters_zygote, ∇obs_zygote, ∇control_zygote = ∇all_zygote;

#=
We can check the results to validate our chain rule.
=#

∇parameters_zygote ≈ ∇parameters_forwarddiff

#-

∇obs_zygote ≈ ∇obs_forwarddiff

#-

∇control_zygote ≈ ∇control_forwarddiff

# ## Reverse mode with Enzyme.jl

#=
The more efficient [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) also works natively as long as there are no type instabilities, which is why we avoid the closure and the keyword arguments with `f_aux`:
=#

function f_aux(parameters, obs_seq, control_seq, seq_ends)
    return f(parameters, obs_seq, control_seq; seq_ends)
end

#=
Enzyme.jl requires preallocated storage for the gradients, which we happily provide.
=#

∇parameters_enzyme = Enzyme.make_zero(parameters)
∇obs_enzyme = Enzyme.make_zero(obs_seq)
∇control_enzyme = Enzyme.make_zero(control_seq);

#=
The syntax is a bit more complex, see the Enzyme.jl docs for details.
=#

try
    Enzyme.autodiff(
        Enzyme.Reverse,
        f_aux,
        Enzyme.Active,
        Enzyme.Duplicated(parameters, ∇parameters_enzyme),
        Enzyme.Duplicated(obs_seq, ∇obs_enzyme),
        Enzyme.Duplicated(control_seq, ∇control_enzyme),
        Enzyme.Const(seq_ends),
    )
catch exception  # latest release of Enzyme broke this code
    display(exception)
end

#=
Once again we can check the results.
=#

∇parameters_enzyme ≈ ∇parameters_forwarddiff

#-

∇obs_enzyme ≈ ∇obs_forwarddiff

#-

∇control_enzyme ≈ ∇control_forwarddiff

#=
For increased efficiency, we could provide temporary storage to Enzyme.jl in order to avoid allocations.
This requires going one level deeper and leveraging the in-place [`HiddenMarkovModels.forward!`](@ref) function.
=#

# ## Gradient methods

#=
Once we have gradients of the loglikelihood, it is a natural idea to perform gradient descent in order to fit the parameters of a custom HMM.
However, there are two caveats we must keep in mind.

First, computing a gradient essentially requires running the forward-backward algorithm, which means it is expensive.
Given the output of forward-backward, if there is a way to perform a more accurate parameter update (like going straight to the maximum likelihood value), it is probably worth it.
That is what we show in the other tutorials with the reimplementation of the `fit!` method.

Second, HMM parameters live in a constrained space, which calls for a projected gradient descent.
Most notably, the transition matrix must be stochastic, and the orthogonal projection onto this set (the Birkhoff polytope) is not easy to obtain.

Still, first order optimization can be relevant when we lack explicit formulas for maximum likelihood.
=#

# ## Tests  #src

@testset "Gradient correctness" begin  #src
    @testset "ForwardDiff" begin  #src
        @test all(!iszero, ∇parameters_forwarddiff)  #src
        @test all(!iszero, ∇obs_forwarddiff)  #src
        @test all(!iszero, ∇control_forwarddiff)  #src
        @test all(isfinite, ∇parameters_forwarddiff)  #src
        @test all(isfinite, ∇obs_forwarddiff)  #src
        @test all(isfinite, ∇control_forwarddiff)  #src
    end  #src
    @testset "Zygote" begin  #src
        @test ∇parameters_zygote ≈ ∇parameters_forwarddiff  #src
        @test ∇obs_zygote ≈ ∇obs_forwarddiff  #src
        @test ∇control_zygote ≈ ∇control_forwarddiff  #src
    end  #src
    @testset "Enzyme" begin  #src
        @test_skip ∇parameters_enzyme ≈ ∇parameters_forwarddiff  #src
        @test_skip ∇obs_enzyme ≈ ∇obs_forwarddiff  #src
        @test_skip ∇control_enzyme ≈ ∇control_forwarddiff  #src
    end  #src
end  #src

control_seqs = [rand(rng, rand(rng, 100:200)) for k in 1:100]  #src
control_seq = reduce(vcat, control_seqs)  #src
seq_ends = cumsum(length.(control_seqs))  #src

test_coherent_algorithms(rng, hmm, control_seq; seq_ends, init=false)  #src
test_type_stability(rng, hmm, control_seq; seq_ends)  #src
