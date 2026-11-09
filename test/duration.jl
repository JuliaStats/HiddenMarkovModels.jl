using HiddenMarkovModels
using HiddenMarkovModels:
    duration_logdensityof,
    duration_logsurvival,
    fit_duration_in_sequence!,
    impute_censored_durations,
    rand_duration
using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using Distributions:
    Distributions,
    Binomial,
    DiscreteUniform,
    Geometric,
    NegativeBinomial,
    Poisson,
    ccdf,
    logccdf,
    params,
    pdf,
    succprob
using Optim: Optim, Brent, optimize
using Random: AbstractRNG
using StableRNGs: StableRNG
using StatsAPI: StatsAPI
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
            exact = logccdf(dist, k - 2)
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

# The duration part of the expected complete-data loglikelihood.
function duration_Q(dist, completed, censored)
    Q = 0.0
    for d in eachindex(completed, censored)
        completed[d] > 0 && (Q += completed[d] * duration_logdensityof(dist, d))
        censored[d] > 0 && (Q += censored[d] * duration_logsurvival(dist, d))
    end
    return Q
end

# Iterate the duration update to its fixed point.
function iterate_duration_fit!(dists, completed, censored; iterations=20_000, rtol=1e-13)
    params_prev = params(dists[1])
    for _ in 1:iterations
        fit_duration_in_sequence!(dists, 1, completed, censored)
        all(isapprox.(params(dists[1]), params_prev; rtol)) && return true
        params_prev = params(dists[1])
    end
    return false
end

# A geometric duration outside Distributions.jl, so it goes through the generic fallback.
mutable struct MutableDuckGeometric
    p::Float64
end
DensityInterface.DensityKind(::MutableDuckGeometric) = HasDensity()
function DensityInterface.logdensityof(d::MutableDuckGeometric, k)
    return k < 0 ? -Inf : log(d.p) + k * log1p(-d.p)
end
function HiddenMarkovModels.duration_logsurvival(d::MutableDuckGeometric, k::Integer)
    return k <= 1 ? 0.0 : (k - 1) * log1p(-d.p)
end
function StatsAPI.fit!(d::MutableDuckGeometric, x::AbstractVector, w::AbstractVector)
    d.p = sum(w) / (sum(w) + sum(w .* x))
    return d
end
Distributions.params(d::MutableDuckGeometric) = (d.p,)

# A duration whose survival never decreases, so its imputed tail never ends.
struct EndlessDuration end
DensityInterface.DensityKind(::EndlessDuration) = HasDensity()
DensityInterface.logdensityof(::EndlessDuration, k) = -1.0
HiddenMarkovModels.duration_logsurvival(::EndlessDuration, k::Integer) = 0.0

@testset "Duration M-step" begin
    completed = [0.0, 3.0, 5.5, 4.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05]
    censored = [0.2, 0.5, 0.3, 0.4, 0.6, 0.1, 0.0, 0.9, 0.0, 0.3]
    D = length(completed)

    @testset "Geometric closed form" begin
        q = sum(completed) / (sum((1:D) .* completed) + sum((0:(D - 1)) .* censored))
        dists = [Geometric(0.5)]
        fit_duration_in_sequence!(dists, 1, completed, censored)
        @test succprob(dists[1]) ≈ q
        res = optimize(
            logit_q -> -duration_Q(Geometric(1 / (1 + exp(-logit_q))), completed, censored),
            -20.0,
            20.0,
            Brent();
            abs_tol=1e-12,
        )
        @test q ≈ 1 / (1 + exp(-Optim.minimizer(res))) rtol = 1e-6
        # Iterating the generic fallback converges to the same maximizer.
        dists_fallback = [MutableDuckGeometric(0.5)]
        @test iterate_duration_fit!(dists_fallback, completed, censored)
        @test dists_fallback[1].p ≈ q rtol = 1e-8

        dists32 = [Geometric(0.5f0)]
        fit_duration_in_sequence!(dists32, 1, Float32.(completed), Float32.(censored))
        @test dists32[1] isa Geometric{Float32}
        @test succprob(dists32[1]) ≈ q rtol = 1e-5

        # If every sojourn lasts one step, stay a valid distribution just below `q = 1`.
        dists1 = [Geometric(0.5)]
        fit_duration_in_sequence!(dists1, 1, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test succprob(dists1[1]) == prevfloat(1.0)
        @test isfinite(duration_logdensityof(dists1[1], 1))
        # Also when `dists` mixes families.
        mixed = [Geometric(0.5), Poisson(2.0)]
        fit_duration_in_sequence!(mixed, 1, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        @test succprob(mixed[1]) == prevfloat(1.0)
    end

    @testset "The imputation fallback maximizes Q" begin
        dists = [Poisson(2.0)]
        Q_evolution = [duration_Q(dists[1], completed, censored)]
        for _ in 1:5
            fit_duration_in_sequence!(dists, 1, completed, censored)
            push!(Q_evolution, duration_Q(dists[1], completed, censored))
        end
        @test all(>=(0), diff(Q_evolution))
        @test iterate_duration_fit!(dists, completed, censored)
        res = optimize(
            logλ -> -duration_Q(Poisson(exp(logλ)), completed, censored),
            log(1e-3),
            log(1e3),
            Brent();
            abs_tol=1e-12,
        )
        @test mean(dists[1]) ≈ exp(Optim.minimizer(res)) rtol = 1e-6
    end

    @testset "Without completed sojourns" begin
        # Fitting refuses rather than running off to an infinite mean.
        @test_throws ArgumentError fit_duration_in_sequence!(
            [Poisson(2.0)], 1, zeros(6), [0.0, 0.0, 1.0, 0.0, 0.5, 0.0]
        )
    end

    @testset "Bounded support" begin
        dist = Binomial(4, 0.5)  # sojourns of 1 to 5 steps
        w = impute_censored_durations(dist, [1.0, 2.0, 0.0], [0.0, 0.0, 1.0])
        @test length(w) == 5
        @test w[1:2] == [1.0, 2.0]
        @test w[3:5] ≈ [pdf(dist, d - 1) / ccdf(dist, 1) for d in 3:5]
        @test sum(w) ≈ 4.0
        # A censored sojourn beyond the support cannot come from the E-step.
        @test_throws ArgumentError impute_censored_durations(
            dist, zeros(7), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        # Unless its weight is rounding noise.
        @test impute_censored_durations(
            dist, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-20]
        ) == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    @testset "Imputed tail" begin
        two_sojourns = ([1.0, 0.0], [0.0, 1.0])
        dist = Geometric(0.1)
        w = impute_censored_durations(dist, two_sojourns...)
        # The tail is followed until the remaining mass is below eps.
        @test sum(w) ≈ 2.0 atol = 1e-13
        @test w[3:10] ≈ [pdf(dist, d - 1) / ccdf(dist, 0) for d in 3:10]
        w32 = impute_censored_durations(
            Geometric(0.1f0), Float32.(two_sojourns[1]), Float32.(two_sojourns[2])
        )
        @test eltype(w32) == Float32
        @test length(w32) < length(w)
        @test sum(w32) ≈ 2.0f0
        # A tiny total, whose `eps` multiple underflows, still ends the tail.
        w_tiny = impute_censored_durations(dist, [1e-310, 0.0], [0.0, 1e-310])
        @test length(w_tiny) <= length(w)
        @test sum(w_tiny) ≈ 2e-310 rtol = 1e-6
        # Never truncated silently.
        @test_throws ArgumentError impute_censored_durations(
            EndlessDuration(), two_sojourns...
        )
    end
end
