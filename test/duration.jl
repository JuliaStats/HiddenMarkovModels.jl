using HiddenMarkovModels
using DensityInterface: logdensityof
using Random
using Statistics
using StatsAPI: fit!
using Test

## duration distributions

# Our minimal duration-distribution interface only requires logdensityof, rand, fit!.
# These tests target support on positive integers, sample mean close to mean implied
# by parameters, and weighted-mean recovery via fit!.

function _expected_mean(d::GeometricDuration)
    return 1 / d.p
end
function _expected_mean(d::PoissonDuration)
    return d.λ + 1
end
function _expected_mean(d::NegBinomialDuration)
    return d.r * (1 - d.p) / d.p + 1
end

function _test_duration_interface(dist::AbstractDurationDistribution, expected_mean::Real)
    @testset "Interface: $(typeof(dist).name.wrapper)" begin
        # Support on positive integers
        @test logdensityof(dist, 0) == -Inf
        @test logdensityof(dist, -1) == -Inf
        for k in 1:20
            @test logdensityof(dist, k) <= 0  # log of a probability
        end

        # Probabilities sum to ~1 over a wide range
        pmf_sum = sum(exp(logdensityof(dist, k)) for k in 1:5000)
        @test pmf_sum ≈ 1.0 atol = 0.01

        # Sampling yields positive integers with the right mean
        rng = MersenneTwister(123)
        samples = [rand(rng, dist) for _ in 1:5000]
        @test all(s isa Integer for s in samples)
        @test all(s >= 1 for s in samples)
        @test mean(samples) ≈ expected_mean rtol = 0.1
    end
end

function _test_duration_fit!(constructor, args, recover_field::Symbol; tol=0.15)
    true_dist = constructor(args...)
    rng = MersenneTwister(42)
    n = 5000
    data = [rand(rng, true_dist) for _ in 1:n]
    weights = ones(n)
    fit_dist = constructor(args...)
    fit!(fit_dist, data, weights)
    @test getfield(fit_dist, recover_field) ≈ getfield(true_dist, recover_field) rtol = tol
end

@testset "Duration distributions" begin
    @testset "GeometricDuration" begin
        @testset "Construction" begin
            d = GeometricDuration(0.5)
            @test d.p == 0.5
            @test_throws Exception GeometricDuration(0.0)
            @test_throws Exception GeometricDuration(1.5)
            @test_throws Exception GeometricDuration(-0.1)
        end

        @testset "Display" begin
            str = sprint(show, GeometricDuration(0.3))
            @test occursin("GeometricDuration", str)
            @test occursin("0.3", str)
        end

        _test_duration_interface(GeometricDuration(0.5), 2.0)
        _test_duration_interface(GeometricDuration(0.25), 4.0)
        _test_duration_interface(GeometricDuration(0.1), 10.0)

        _test_duration_fit!(GeometricDuration, (0.5,), :p)
        _test_duration_fit!(GeometricDuration, (0.25,), :p)
    end

    @testset "PoissonDuration" begin
        @testset "Construction" begin
            d = PoissonDuration(3.0)
            @test d.λ == 3.0
            @test_throws Exception PoissonDuration(0.0)
            @test_throws Exception PoissonDuration(-1.0)
        end

        @testset "Display" begin
            str = sprint(show, PoissonDuration(5.0))
            @test occursin("PoissonDuration", str)
            @test occursin("5.0", str)
        end

        _test_duration_interface(PoissonDuration(3.0), 4.0)
        _test_duration_interface(PoissonDuration(5.0), 6.0)

        _test_duration_fit!(PoissonDuration, (3.0,), :λ)
        _test_duration_fit!(PoissonDuration, (5.0,), :λ)
    end

    @testset "NegBinomialDuration" begin
        @testset "Construction" begin
            d = NegBinomialDuration(5.0, 0.5)
            @test d.r == 5.0
            @test d.p == 0.5
            @test NegBinomialDuration(5, 0.5).r isa AbstractFloat  # promotion
            @test_throws Exception NegBinomialDuration(-1.0, 0.5)
            @test_throws Exception NegBinomialDuration(5.0, 0.0)
            @test_throws Exception NegBinomialDuration(5.0, 1.0)
        end

        @testset "Display" begin
            str = sprint(show, NegBinomialDuration(3.0, 0.7))
            @test occursin("NegBinomialDuration", str)
            @test occursin("3.0", str)
            @test occursin("0.7", str)
        end

        _test_duration_interface(NegBinomialDuration(5.0, 0.5), 5.0 * (1 - 0.5) / 0.5 + 1)

        _test_duration_fit!(NegBinomialDuration, (5.0, 0.5), :r; tol=0.3)
    end

    @testset "Distributions are distinct" begin
        geom = GeometricDuration(0.5)
        pois = PoissonDuration(1.0)  # both have mean 2
        @test any(
            abs(exp(logdensityof(geom, k)) - exp(logdensityof(pois, k))) > 1e-6 for
            k in 2:10
        )
    end
end
