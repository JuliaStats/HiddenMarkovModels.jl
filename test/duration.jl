using HiddenMarkovModels
using HiddenMarkovModels: duration_logdensityof, duration_logsurvival, rand_duration
using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using Distributions: DiscreteUniform, Geometric, NegativeBinomial, Poisson, logccdf
using Random: AbstractRNG
using StableRNGs: StableRNG
using Statistics: mean
using Test

# A minimal user-defined duration distribution that only implements the required interface,
# to confirm we never reach into `Distributions`-specific internals.
struct ConstDuration{T} end
DensityInterface.DensityKind(::ConstDuration) = HasDensity()
DensityInterface.logdensityof(::ConstDuration, x) = iszero(x) ? 0.0 : -Inf
Base.rand(::AbstractRNG, ::ConstDuration{T}) where {T} = zero(T)
HiddenMarkovModels.duration_logsurvival(::ConstDuration, k::Integer) = k <= 1 ? 0.0 : -Inf

# Wraps a `Distributions` law without being one, so it must supply its own survival function.
# It uses the two-line summation suggested in the `duration_logsurvival` docstring.
struct DuckDuration{D}
    dist::D
end
DensityInterface.DensityKind(::DuckDuration) = HasDensity()
DensityInterface.logdensityof(d::DuckDuration, x) = logdensityof(d.dist, x)
Base.rand(rng::AbstractRNG, d::DuckDuration) = rand(rng, d.dist)
function HiddenMarkovModels.duration_logsurvival(d::DuckDuration, k::Integer)
    k <= 1 && return 0.0
    return log1p(-sum(exp(duration_logdensityof(d, j)) for j in 1:(k - 1)))
end

# Implements the density but forgets the survival function.
struct ForgetfulDuration end
DensityInterface.DensityKind(::ForgetfulDuration) = HasDensity()
DensityInterface.logdensityof(::ForgetfulDuration, x) = iszero(x) ? 0.0 : -Inf

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

    @testset "Sample type is preserved" begin
        # The `+1` shift must keep the sampled type.
        for T in (Int, Int32, Float32)
            d = ConstDuration{T}()
            @test rand_duration(StableRNG(1), d) === one(T)
            @test rand_duration(d) === one(T)  # default rng overload
        end
    end

    @testset "Custom user distribution (interface only)" begin
        d = ConstDuration{Int}()
        @test duration_logdensityof(d, 1) == 0.0
        @test duration_logdensityof(d, 2) == -Inf
        @test duration_logdensityof(d, 0) == -Inf
        @test rand_duration(StableRNG(1), d) == 1
        @test rand_duration(d) == 1  # default rng overload
    end
end

@testset "Duration survival" begin
    dists = (Poisson(3.0), Geometric(0.5), Geometric(0.05), NegativeBinomial(3.0, 0.4))

    @testset "Agrees with the closed form far into the tail" begin
        for dist in dists, k in 1:80
            exact = k <= 1 ? 0.0 : logccdf(dist, k - 2)
            @test duration_logsurvival(dist, k) ≈ exact atol = 1e-10
        end
    end

    @testset "Every sojourn lasts at least one timestep" begin
        for dist in (dists..., Geometric(1.0), DiscreteUniform(0, 5))
            @test duration_logsurvival(dist, 1) == 0
            @test duration_logsurvival(dist, 0) == 0
        end
    end

    @testset "Bounded support runs out" begin
        d = DiscreteUniform(0, 5)
        @test duration_logsurvival(d, 6) ≈ log(1 / 6)
        @test duration_logsurvival(d, 7) == -Inf
        @test duration_logsurvival(d, 50) == -Inf
    end

    @testset "Degenerate law" begin
        @test duration_logsurvival(Geometric(1.0), 1) == 0
        @test duration_logsurvival(Geometric(1.0), 2) == -Inf
    end

    @testset "Consistent with the duration density" begin
        for dist in dists, k in 1:15
            s1, s2 = duration_logsurvival(dist, k), duration_logsurvival(dist, k + 1)
            @test exp(s1) - exp(s2) ≈ exp(duration_logdensityof(dist, k)) atol = 1e-12
        end
    end

    @testset "Custom user distribution (interface only)" begin
        @test duration_logsurvival(ConstDuration{Int}(), 1) == 0
        @test duration_logsurvival(ConstDuration{Int}(), 2) == -Inf

        # The documented summation recipe agrees with the closed form.
        # `log1p(-head)` loses precision as `head -> 1`, hence the relative tolerance.
        for dist in dists, k in 1:20
            @test duration_logsurvival(DuckDuration(dist), k) ≈
                duration_logsurvival(dist, k) atol = 1e-8 rtol = 1e-6
        end
    end

    @testset "No silent fallback" begin
        @test_throws MethodError duration_logsurvival(ForgetfulDuration(), 2)
    end
end
