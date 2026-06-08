using HiddenMarkovModels
using HiddenMarkovModels: duration_logdensityof, rand_duration
using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using Distributions: Geometric, NegativeBinomial, Poisson
using Random: AbstractRNG
using StableRNGs: StableRNG
using Statistics: mean
using Test

# A minimal user-defined duration distribution that only implements the required interface,
# to confirm we never reach into `Distributions`-specific internals.
struct ShiftedConst end  # deterministic: (sojourn - 1) == 0 always, i.e. sojourn == 1
DensityInterface.DensityKind(::ShiftedConst) = HasDensity()
DensityInterface.logdensityof(::ShiftedConst, x) = iszero(x) ? 0.0 : -Inf
Base.rand(::AbstractRNG, ::ShiftedConst) = 0

@testset "Duration convention" begin
    @testset "Shift relationship" begin
        for dist in (Poisson(2.0), Geometric(0.3), NegativeBinomial(4.0, 0.5))
            for k in -2:20
                @test duration_logdensityof(dist, k) == logdensityof(dist, k - 1)
            end
            # Support is pushed onto the strictly positive integers.
            @test duration_logdensityof(dist, 0) == -Inf
            @test duration_logdensityof(dist, -1) == -Inf
            @test duration_logdensityof(dist, 1) > -Inf
        end
    end

    @testset "Normalization over sojourn times" begin
        for dist in (Poisson(2.0), Geometric(0.3), NegativeBinomial(4.0, 0.5))
            pmf_sum = sum(exp(duration_logdensityof(dist, k)) for k in 1:5000)
            @test pmf_sum ≈ 1.0 atol = 1e-6
        end
    end

    @testset "Sampling is shifted by one" begin
        rng = StableRNG(63)
        # mean(dist) + 1, since the sojourn is (the sampled value) + 1.
        for (dist, expected_mean) in (
            (Poisson(2.0), 3.0),        # mean 2 -> 3
            (Geometric(0.25), 4.0),     # mean 3 -> 4
            (NegativeBinomial(4.0, 0.5), 5.0),  # mean 4 -> 5
        )
            samples = [rand_duration(rng, dist) for _ in 1:10_000]
            @test all(s isa Integer for s in samples)
            @test all(s >= 1 for s in samples)
            @test mean(samples) ≈ expected_mean rtol = 0.05
        end
    end

    @testset "Custom user distribution (interface only)" begin
        d = ShiftedConst()
        @test duration_logdensityof(d, 1) == 0.0
        @test duration_logdensityof(d, 2) == -Inf
        @test duration_logdensityof(d, 0) == -Inf
        @test rand_duration(StableRNG(1), d) == 1
        @test rand_duration(d) == 1  # default rng overload
    end
end
