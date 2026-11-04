"""
$(SIGNATURES)

Log-probability that a state with sojourn-time distribution `dist` is held for exactly `k`
consecutive timesteps.

`dist` is supplied by the user with support on `{0, 1, 2, ...}` and is interpreted as the law
of `(sojourn time - 1)`. The sojourn time itself lives on `{1, 2, 3, ...}`, so this returns
`logdensityof(dist, k - 1)`. Durations `k < 1` fall outside the user's support and yield
`-Inf` through `dist` (e.g. a `Distributions.Distribution` returns `-Inf` for a negative
argument).

See also [`rand_duration`](@ref) and [`duration_logsurvival`](@ref).
"""
duration_logdensityof(dist, k::Integer) = logdensityof(dist, k - one(k))

"""
$(SIGNATURES)

Sample a sojourn time on `{1, 2, 3, ...}` from the duration distribution `dist`.

`dist` is interpreted as the law of `(sojourn time - 1)` (see [`duration_logdensityof`](@ref)),
so this returns `rand(rng, dist) + 1`.
"""
function rand_duration(rng::AbstractRNG, dist)
    t = rand(rng, dist)
    return t + one(t)
end

rand_duration(dist) = rand_duration(default_rng(), dist)

"""
    duration_logsurvival(dist, k)

Return `log ℙ(sojourn time >= k)` for the duration distribution `dist`, following the
convention of [`duration_logdensityof`](@ref) (the sojourn time lives on `{1, 2, 3, ...}`).

This is the right-censoring term for the final segment of a sequence, whose sojourn is only
known to last at least `k` timesteps. Since every sojourn lasts at least one timestep, the
result must be `0 = log(1)` for `k <= 1`.

A closed-form method based on `logccdf` is provided for every
`Distributions.DiscreteUnivariateDistribution` when Distributions.jl is loaded. Custom duration
distributions must implement this function themselves, for instance as

```julia
function HiddenMarkovModels.duration_logsurvival(dist::MyDuration, k::Integer)
    k <= 1 && return 0.0
    return log1p(-sum(exp(duration_logdensityof(dist, j)) for j in 1:(k - 1)))
end
```

(but taking more precautions with numerical instabilities).
"""
function duration_logsurvival end

"""
    fit_duration_in_sequence!(dists, i, completed, censored)

Modify the `i`-th element of `dists`, a vector of duration distributions, to maximize

    Q(θ) = Σ_d completed[d] * log p(d; θ) + Σ_d censored[d] * log S(d; θ)

where `p` and `S` are given by [`duration_logdensityof`](@ref) and
[`duration_logsurvival`](@ref). `completed[d]` and `censored[d]` are the expected numbers of
completed sojourns lasting exactly `d` steps, and of right-censored final sojourns still running
after `d` steps, i.e. the expected duration counts of the Baum-Welch E-step.

A method must keep `dists[i]` in the same family and must not decrease `Q` (maximizing it is
best). Baum-Welch only calls it with `sum(completed) > 0`.

Provided methods:

- A closed form for `Distributions.Geometric`.
- A generic fallback, which imputes the remainder of the censored sojourns under the current
  `dists[i]`, then calls the weighted fit [`HiddenMarkovModels.fit_in_sequence!`](@ref) on the
  durations `0:(L-1)` (the distributions model `sojourn - 1`). Without a weighted fit, its error
  propagates: a `MethodError` for `StatsAPI.fit!`, or for Distributions.jl, an error from
  `suffstats` (e.g. for `NegativeBinomial`). Distributions.jl only provides weighted fits with
  `Float64` weights.
"""
function fit_duration_in_sequence!(
    dists::AbstractVector, i::Integer, completed::AbstractVector, censored::AbstractVector
)
    return fit_duration_by_imputation!(dists, i, completed, censored)
end

function check_completed_sojourns(completed::AbstractVector)
    @argcheck sum(completed) > zero(eltype(completed)) "Durations cannot be fitted without completed sojourns."
    return nothing
end

# Bounds the work spent imputing the tail of a single state (2^20 weights, 8 MiB in Float64).
const DEFAULT_MAX_IMPUTED_DURATION = 2^20

#=
Return the weights `w[d]`, `d = 1, ..., L`, of the sojourn lengths once the right-censored
sojourns are completed under `dist`: each `censored[d]` is spread over `d' ≥ d` proportionally to
`p(d') / S(d)`, and added to `completed`. `L` grows until the imputed mass beyond it is below
`eps` times the total weight, or the support of `dist` ends.
=#
function impute_censored_durations(
    dist,
    completed::AbstractVector,
    censored::AbstractVector;
    max_imputed_duration::Integer=DEFAULT_MAX_IMPUTED_DURATION,
)
    @argcheck length(completed) == length(censored)
    R = float(promote_type(eltype(completed), eltype(censored)))
    weights = Vector{R}(undef, length(completed))
    copyto!(weights, completed)
    total = sum(weights) + convert(R, sum(censored))
    log_zero = convert(R, -Inf)
    # Weights below this are lost to rounding when added to the total.
    negligible = eps(R) * total
    # `log_spread = log Σ_{d ≤ d'} censored[d] / S(d)`, so that `d'` gets `p(d') * exp(log_spread)`.
    log_spread = log_zero
    for (d, c) in enumerate(censored)
        if c > zero(c)
            log_surv = convert(R, duration_logsurvival(dist, d))
            check_not_nan(log_surv, dist, d, "duration_logsurvival")
            if is_log_zero(log_surv)
                # Only rounding noise can give weight to an impossible censored sojourn.
                @argcheck c <= negligible "Censored sojourns of length $d have zero survival under $dist."
            else
                log_spread = logaddexp_safe(log_spread, log(convert(R, c)) - log_surv)
            end
        end
        add_imputed_weight!(weights, d, dist, log_spread)
    end
    is_log_zero(log_spread) && return weights
    log_negligible = log(negligible)
    L = length(weights)
    while true
        # The mass left beyond `L` is `S(L + 1) * exp(log_spread)`, zero after the support.
        log_surv = convert(R, duration_logsurvival(dist, L + 1))
        check_not_nan(log_surv, dist, L + 1, "duration_logsurvival")
        log_spread + log_surv <= log_negligible && break
        @argcheck L < max_imputed_duration "The imputed tail of censored sojourns under $dist exceeds $max_imputed_duration durations."
        L += 1
        push!(weights, zero(R))
        add_imputed_weight!(weights, L, dist, log_spread)
    end
    return weights
end

function add_imputed_weight!(
    weights::AbstractVector{R}, d::Integer, dist, log_spread::R
) where {R}
    is_log_zero(log_spread) && return nothing
    log_p = convert(R, duration_logdensityof(dist, d))
    check_not_nan(log_p, dist, d, "duration_logdensityof")
    is_log_zero(log_p) || (weights[d] += exp(log_p + log_spread))
    return nothing
end

function check_not_nan(x, dist, d::Integer, name::String)
    @argcheck !isnan(x) "`$name` returned NaN for $dist at duration $d."
    return nothing
end

#=
Generic `fit_duration_in_sequence!`: an exact EM step for `Q`, treating the unobserved remainder
of the censored sojourns as missing data, see the formulas page of the documentation.
=#
function fit_duration_by_imputation!(
    dists::AbstractVector,
    i::Integer,
    completed::AbstractVector,
    censored::AbstractVector;
    max_imputed_duration::Integer=DEFAULT_MAX_IMPUTED_DURATION,
)
    check_completed_sojourns(completed)
    weights = impute_censored_durations(dists[i], completed, censored; max_imputed_duration)
    fit_in_sequence!(dists, i, 0:(length(weights) - 1), weights)
    return nothing
end
