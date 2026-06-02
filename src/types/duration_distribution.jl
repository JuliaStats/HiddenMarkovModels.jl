"""
    AbstractDurationDistribution

Abstract supertype for per-state sojourn-time (duration) distributions, as used by hidden
semi-Markov models. The support is the positive integers `{1, 2, 3, ...}` — a duration of `k`
means the state is held for `k` consecutive timesteps before transitioning.

# Interface

To implement your own duration distribution, define:

- `DensityInterface.logdensityof(d, k::Int)` — `log P(D = k)`
- `Random.rand(rng, d)` — sample a positive-integer duration
- `StatsAPI.fit!(d, durations::AbstractVector{Int}, weights::AbstractVector)` — refit from
  weighted sojourn samples (used by Baum-Welch)
"""
abstract type AbstractDurationDistribution end

@inline DensityInterface.DensityKind(::AbstractDurationDistribution) = HasDensity()

"""
    GeometricDuration(p) <: AbstractDurationDistribution

Shifted geometric duration on `{1, 2, 3, ...}` (equivalent to `Geometric(p) + 1` from `Distributions.jl`).

`p` is the per-step probability of leaving the state. Mean duration is `1/p`.

Geometric durations correspond to the implicit sojourn distribution of a standard HMM, which makes
this the natural choice when migrating an existing HMM model to an HSMM.
"""
mutable struct GeometricDuration{T<:Real} <: AbstractDurationDistribution
    "per-step exit probability, `0 < p ≤ 1`"
    p::T

    function GeometricDuration(p::T) where {T<:Real}
        @argcheck 0 < p <= 1 "Exit probability must be in (0, 1]"
        return new{T}(p)
    end
end

Base.show(io::IO, d::GeometricDuration) = print(io, "GeometricDuration(p=$(d.p))")

function DensityInterface.logdensityof(d::GeometricDuration, k::Int)
    return k >= 1 ? log(d.p) + (k - 1) * log1p(-d.p) : oftype(log(d.p), -Inf)
end

function Random.rand(rng::AbstractRNG, d::GeometricDuration)
    # Inverse-CDF sampling of shifted geometric on {1, 2, ...}:
    # If U ~ Uniform(0,1), then 1 + floor(log(U) / log(1-p)) ~ Geometric(p) + 1.
    u = rand(rng)
    return 1 + floor(Int, log(u) / log1p(-d.p))
end

function StatsAPI.fit!(
    d::GeometricDuration, durations::AbstractVector{Int}, weights::AbstractVector
)
    s = zero(eltype(weights))
    sw = zero(eltype(weights))
    @simd for i in eachindex(durations, weights)
        s += durations[i] * weights[i]
        sw += weights[i]
    end
    new_p = sw / s  # 1 / weighted_mean
    d.p = clamp(new_p, 1e-10, one(d.p))
    return nothing
end

# Log-space Knuth: stable for λ values where exp(-λ) would underflow. O(λ) expected time
# is acceptable for HSMM sojourns; durations rarely have λ > 100.
function _poisson_rand(rng::AbstractRNG, λ::Real)
    target = -float(λ)
    log_p = zero(target)
    k = 0
    while true
        k += 1
        log_p += log(rand(rng))
        if log_p <= target
            return k - 1
        end
    end
end

# Marsaglia & Tsang's Gamma(α, 1) sampler. For α < 1, use the boost identity
# Gamma(α) =d= Gamma(α+1) · U^(1/α) (Stuart 1962) so the inner sampler only runs at α ≥ 1.
function _gamma_rand(rng::AbstractRNG, α::Real)
    if α < 1
        return _gamma_rand(rng, α + 1) * rand(rng)^(1 / α)
    end
    d = α - 1 / 3
    c = 1 / sqrt(9 * d)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        if v > 0
            u = rand(rng)
            if log(u) < x^2 / 2 + d - d * v + d * log(v)
                return d * v
            end
        end
    end
end

"""
    PoissonDuration(λ) <: AbstractDurationDistribution

Shifted Poisson duration on `{1, 2, 3, ...}` (equivalent to `Poisson(λ) + 1` from `Distributions.jl`).

Mean duration is `λ + 1`. Common choice when state durations cluster around a mode larger than one.
"""
mutable struct PoissonDuration{T<:Real} <: AbstractDurationDistribution
    "rate parameter, `λ > 0`"
    λ::T

    function PoissonDuration(λ::T) where {T<:Real}
        @argcheck λ > 0 "Rate parameter must be positive"
        return new{T}(λ)
    end
end

Base.show(io::IO, d::PoissonDuration) = print(io, "PoissonDuration(λ=$(d.λ))")

function DensityInterface.logdensityof(d::PoissonDuration, k::Int)
    return k >= 1 ? poislogpdf(d.λ, k - 1) : oftype(float(d.λ), -Inf)
end

Random.rand(rng::AbstractRNG, d::PoissonDuration) = _poisson_rand(rng, d.λ) + 1

# Poisson MLE equals method-of-moments here: λ̂ = weighted mean of shifted durations.
function StatsAPI.fit!(
    d::PoissonDuration, durations::AbstractVector{Int}, weights::AbstractVector
)
    s = zero(eltype(weights))
    sw = zero(eltype(weights))
    @simd for i in eachindex(durations, weights)
        s += durations[i] * weights[i]
        sw += weights[i]
    end
    d.λ = max(oftype(d.λ, s / sw - 1), 1e-10)
    return nothing
end

"""
    NegBinomialDuration(r, p) <: AbstractDurationDistribution

Shifted negative-binomial duration on `{1, 2, 3, ...}` (equivalent to `NegativeBinomial(r, p) + 1`).

Mean duration is `r(1-p)/p + 1`, variance is `r(1-p)/p²`. Useful when sojourn lengths are
overdispersed relative to a Poisson.
"""
mutable struct NegBinomialDuration{T<:Real} <: AbstractDurationDistribution
    "number of successes, `r > 0`"
    r::T
    "success probability, `0 < p < 1`"
    p::T

    function NegBinomialDuration(r::Real, p::Real)
        r_promoted, p_promoted = promote(r, p)
        @argcheck r_promoted > 0 "Number of successes must be positive"
        @argcheck 0 < p_promoted < 1 "Success probability must be in (0, 1)"
        return new{typeof(r_promoted)}(r_promoted, p_promoted)
    end
end

function Base.show(io::IO, d::NegBinomialDuration)
    return print(io, "NegBinomialDuration(r=$(d.r), p=$(d.p))")
end

function DensityInterface.logdensityof(d::NegBinomialDuration, k::Int)
    return k >= 1 ? nbinomlogpdf(d.r, d.p, k - 1) : oftype(float(d.r), -Inf)
end

function Random.rand(rng::AbstractRNG, d::NegBinomialDuration)
    # NB(r, p) =d= Poisson(λ) with λ ~ Gamma(shape=r, scale=(1-p)/p).
    scale = (1 - d.p) / d.p
    λ_sample = _gamma_rand(rng, d.r) * scale
    return _poisson_rand(rng, λ_sample) + 1
end

# MLE for NegBinomial(r, p) on the shifted samples. Profile out p* = r W / (r W + S),
# leaving a 1-D Newton step on r driven by the digamma score equation; trigamma supplies
# the second derivative analytically. Falls back to a Poisson-like estimator when the
# sample is underdispersed and the MLE has no interior solution.
function StatsAPI.fit!(
    d::NegBinomialDuration, durations::AbstractVector{Int}, weights::AbstractVector
)
    W = zero(eltype(weights))
    S = zero(eltype(weights))
    @simd for i in eachindex(durations, weights)
        W += weights[i]
        S += (durations[i] - 1) * weights[i]
    end
    mean_x = S / W

    S2 = zero(eltype(weights))
    @simd for i in eachindex(durations, weights)
        S2 += ((durations[i] - 1) - mean_x)^2 * weights[i]
    end
    var_x = S2 / W

    if var_x <= mean_x || mean_x <= 0
        # Underdispersed MLE pushes r → ∞. Pick something stable.
        d.p = oftype(d.p, 0.5)
        d.r = oftype(d.r, max(2 * mean_x, 1e-10))
        return nothing
    end

    # Method-of-moments seed.
    r = mean_x^2 / (var_x - mean_x)
    logW = log(W)

    for _ in 1:100
        sum_psi = zero(r)
        sum_trig = zero(r)
        @simd for i in eachindex(durations, weights)
            k = durations[i] - 1
            sum_psi += weights[i] * digamma(k + r)
            sum_trig += weights[i] * trigamma(k + r)
        end

        # g(r) = ∂ℓ_profile/∂r = Σ ψ(k+r) - Wψ(r) + W log p*(r), and
        # log p*(r) = log(Wr/(Wr+S)) = log W + log r - log(Wr+S).
        g = sum_psi - W * digamma(r) + W * log(r) + W * logW - W * log(W * r + S)
        g_prime = sum_trig - W * trigamma(r) + W / r - (W * W) / (W * r + S)

        abs(g_prime) < 1e-15 && break

        Δ = g / g_prime
        r_new = r - Δ
        # Backtrack if Newton overshoots into infeasible region.
        if r_new <= 0
            r_new = r / 2
        end
        if abs(r_new - r) < 1e-10 * max(r, 1.0)
            r = r_new
            break
        end
        r = r_new
    end

    p_est = W * r / (W * r + S)
    d.p = clamp(oftype(d.p, p_est), 1e-10, 1 - 1e-10)
    d.r = max(oftype(d.r, r), 1e-10)
    return nothing
end
