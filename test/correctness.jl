using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: LightDiagNormal, LightCategorical
using HMMTest
using LinearAlgebra
using Random: Random, AbstractRNG, default_rng, seed!
using SparseArrays
using StableRNGs
using Test

TEST_SUITE = get(ENV, "JULIA_HMM_TEST_SUITE", "Standard")

## Settings

T, K = 100, 200

init = [0.4, 0.6]
init_guess = [0.5, 0.5]

trans = [0.7 0.3; 0.3 0.7]
trans_guess = [0.6 0.4; 0.4 0.6]

p = [[0.8, 0.2], [0.2, 0.8]]
p_guess = [[0.7, 0.3], [0.3, 0.7]]

μ = [-ones(2), +ones(2)]
μ_guess = [-0.8 * ones(2), +0.8 * ones(2)]

σ = ones(2)

rng = StableRNG(63)
control_seqs = [fill(nothing, rand(rng, T:(2T))) for k in 1:K];
control_seq = reduce(vcat, control_seqs);
seq_ends = cumsum(length.(control_seqs));

## Uncontrolled

@testset verbose = true "Normal" begin
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "HMMBase"
        test_identical_hmmbase(rng, hmm, T; hmm_guess)
    else
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
        test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "DiagNormal" begin
    dists = [MvNormal(μ[1], Diagonal(abs2.(σ))), MvNormal(μ[2], Diagonal(abs2.(σ)))]
    dists_guess = [
        MvNormal(μ_guess[1], Diagonal(abs2.(σ))), MvNormal(μ_guess[2], Diagonal(abs2.(σ)))
    ]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "HMMBase"
        test_identical_hmmbase(rng, hmm, T; hmm_guess)
    else
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "LightCategorical" begin
    dists = [LightCategorical(p[1]), LightCategorical(p[2])]
    dists_guess = [LightCategorical(p_guess[1]), LightCategorical(p_guess[2])]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE != "HMMBase"
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
        test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "LightDiagNormal" begin
    dists = [LightDiagNormal(μ[1], σ), LightDiagNormal(μ[2], σ)]
    dists_guess = [LightDiagNormal(μ_guess[1], σ), LightDiagNormal(μ_guess[2], σ)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE != "HMMBase"
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
        test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "Normal (sparse)" begin
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]

    hmm = HMM(init, sparse(trans), dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "HMMBase"
        test_identical_hmmbase(rng, hmm, T; hmm_guess)
    else
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
        @test_skip test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "Normal transposed" begin  # issue 99
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]

    hmm = transpose_hmm(HMM(init, trans, dists))
    hmm_guess = transpose_hmm(HMM(init_guess, trans_guess, dists_guess))

    rng = StableRNG(63)
    if TEST_SUITE == "HMMBase"
        test_identical_hmmbase(rng, hmm, T; hmm_guess)
    else
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
        test_type_stability(rng, hmm, control_seq; seq_ends, hmm_guess)
        test_allocations(rng, hmm, control_seq; seq_ends, hmm_guess)
    end
end

@testset verbose = true "Normal and Exponential" begin  # issue 101
    dists = [Normal(μ[1][1]), Exponential(1.0)]
    dists_guess = [Normal(μ_guess[1][1]), Exponential(0.8)]

    hmm = HMM(init, trans, dists)
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "HMMBase"
        test_identical_hmmbase(rng, hmm, T; hmm_guess)
    else
        test_coherent_algorithms(rng, hmm, control_seq; seq_ends, hmm_guess, init=false)
    end
end

if TEST_SUITE != "HMMBase"
    @testset verbose = true "GaussianGLM - Controlled HMM" begin
        using DensityInterface
        using StatsAPI

        mutable struct GLMNormalModel{T} <: ControlledEmission
            β0::T
            β1::T
            logσ::T
        end

        # Control-aware logdensity: accept any Real types and promote
        function DensityInterface.logdensityof(
            mod::GLMNormalModel, obs::Real, control::Real
        )
            T = promote_type(typeof(mod.β0), typeof(obs), typeof(control))
            μ = T(mod.β0) + T(mod.β1) * T(control)
            s = exp(T(mod.logσ))
            z = (T(obs) - μ) / s
            return -T(0.5) * log(T(2) * T(pi)) - log(s) - T(0.5) * z * z
        end

        # Control-aware sampling: accept any Real control
        function Random.rand(rng::AbstractRNG, mod::GLMNormalModel, control::Real)
            T = promote_type(typeof(mod.β0), typeof(control))
            μ = T(mod.β0) + T(mod.β1) * T(control)
            s = exp(T(mod.logσ))
            return μ + s * randn(rng, T)
        end

        function StatsAPI.fit!(
            mod::GLMNormalModel,
            data::AbstractVector{<:Real},
            control_seq::AbstractVector{<:Real},
            weights::AbstractVector{<:Real},
        )
            n = length(data)
            @assert length(control_seq) == n
            @assert length(weights) == n

            T = promote_type(
                typeof(mod.β0), eltype(data), eltype(control_seq), eltype(weights)
            )

            S0 = zero(T)  # Σ w
            S1 = zero(T)  # Σ w x
            S2 = zero(T)  # Σ w x^2
            T0 = zero(T)  # Σ w y
            T1 = zero(T)  # Σ w x y

            for i in 1:n
                wi = T(weights[i])
                xi = T(control_seq[i])
                yi = T(data[i])
                S0 += wi
                S1 += wi * xi
                S2 += wi * xi * xi
                T0 += wi * yi
                T1 += wi * xi * yi
            end

            D = S0 * S2 - S1 * S1

            β0 = (T0 * S2 - T1 * S1) / D
            β1 = (T1 * S0 - T0 * S1) / D

            sse = zero(T)
            for i in 1:n
                wi = T(weights[i])
                xi = T(control_seq[i])
                yi = T(data[i])
                r = yi - (β0 + β1 * xi)
                sse += wi * r * r
            end

            σ = sqrt(sse / S0)

            mod.β0 = β0
            mod.β1 = β1
            mod.logσ = log(σ)
            return nothing
        end

        #=
        create a true model and a guess model
        =#
        dists = [GLMNormalModel(-1.0, 2.0, log(0.5)), GLMNormalModel(0.0, -1.0, log(1.0))]

        dists_guess = [
            GLMNormalModel(-0.5, 1.0, log(1.0)), GLMNormalModel(0.0, 0.0, log(1.0))
        ]

        hmm_true = ControlledEmissionHMM(init, trans, dists)
        hmm_guess = ControlledEmissionHMM(init_guess, trans_guess, dists_guess)

        Tmin, Tmax = 50, 250
        lens = rand(rng, Tmin:Tmax, K)
        seq_ends_glm = cumsum(lens)

        control_seq_glm = rand(rng, Uniform(-5.0, 5.0), sum(lens))
        test_coherent_algorithms(
            rng,
            hmm_true,
            control_seq_glm;
            seq_ends=seq_ends_glm,
            hmm_guess=hmm_guess,
            init=false,
        )
        test_type_stability(
            rng, hmm_true, control_seq_glm; seq_ends=seq_ends_glm, hmm_guess=hmm_guess
        )
        test_allocations(
            rng, hmm_true, control_seq_glm; seq_ends=seq_ends_glm, hmm_guess=hmm_guess
        )
    end
end
