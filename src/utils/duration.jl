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

where `p(d; θ) = exp(duration_logdensityof(dist, d))` and `S(d; θ) = exp(duration_logsurvival(dist, d))`
for a distribution `dist` of the same family as `dists[i]`.

The vectors `completed` and `censored` are indexed by the sojourn length `d = 1, 2, ...` and hold
the expected numbers of completed sojourns lasting exactly `d` steps, and of right-censored final
sojourns that were still running after `d` steps. This is the duration part of the M-step of
[`baum_welch`](@ref) for an HSMM, where these counts come from [`forward_backward!`](@ref) with
`duration_counts=true`.

# Contract

A method must leave `dists[i]` in the same family (so that `dists` keeps its element type), and
must not decrease `Q` with respect to the current `dists[i]`. Maximizing `Q` exactly is best,
but any increase keeps the loglikelihood of Baum-Welch non-decreasing (generalized EM).
The current `dists[i]` is the distribution used in the E-step, so a method may rely on it, for
instance as a starting point.

[`fit!`](@ref) guarantees that `sum(completed) > 0`, and keeps the duration distribution of a
state without completed sojourns unchanged instead of calling this function. With censored
sojourns only, the data give no upper bound on the durations, so the maximizer of `Q` runs off to
an infinite mean, while keeping the distribution keeps `Q` (and the loglikelihood)
non-decreasing. The provided methods throw an `ArgumentError` if called without completed
sojourns.

# Provided methods

- A closed form for `Distributions.Geometric`, when Distributions.jl is loaded.
- For any other type, an exact EM step that imputes the unobserved remainder of the censored
  sojourns under the current `dists[i]`, then calls the weighted fit
  [`HiddenMarkovModels.fit_in_sequence!`](@ref) on the shifted durations `0:(L-1)` (the
  distributions model `sojourn - 1`). See [`HiddenMarkovModels.fit_duration_by_imputation!`](@ref).

The fallback requires a weighted fit, i.e. `StatsAPI.fit!(dist, x, w)`, a method of
`fit_in_sequence!` for the vector of distributions, or, for Distributions.jl, a weighted
`fit(typeof(dist), x, w)` (not available for e.g. `NegativeBinomial` or `Binomial`). For duration
types without one, or to provide an exact closed form, implement

```julia
function HiddenMarkovModels.fit_duration_in_sequence!(
    dists::AbstractVector{<:MyDuration}, i::Integer, completed::AbstractVector, censored::AbstractVector
)
    dists[i] = ...  # maximize Q, or at least do not decrease it
    return nothing
end
```
"""
function fit_duration_in_sequence!(
    dists::AbstractVector, i::Integer, completed::AbstractVector, censored::AbstractVector
)
    return fit_duration_by_imputation!(dists, i, completed, censored)
end

function check_completed_sojourns(completed::AbstractVector)
    sum(completed) > zero(eltype(completed)) || throw(
        ArgumentError(
            "The duration distribution of a state without completed sojourns cannot be " *
            "fitted, since its durations have no upper bound. `fit!` keeps it unchanged.",
        ),
    )
    return nothing
end

#= The cap bounds the work and memory (2^20 weights, 8 MiB in Float64) spent on a single state.
The imputation only needs that many steps when the current distribution still puts a relative
mass of `eps` beyond them, i.e. with sojourns of order 10^4 steps or longer, where a custom
`fit_duration_in_sequence!` is the better choice anyway. =#
const DEFAULT_MAX_IMPUTED_DURATION = 2^20

"""
    DurationFitError

Thrown by [`HiddenMarkovModels.fit_duration_by_imputation!`](@ref) when the type of a duration
distribution has no weighted fit for [`HiddenMarkovModels.fit_in_sequence!`](@ref).
"""
struct DurationFitError <: Exception
    dist_type::Type
end

function Base.showerror(io::IO, err::DurationFitError)
    return print(
        io,
        "DurationFitError: duration distributions of type $(err.dist_type) have no weighted ",
        "fit, which the default `HiddenMarkovModels.fit_duration_in_sequence!` needs. Either ",
        "provide one (`StatsAPI.fit!(dist, x, w)`, or `fit(typeof(dist), x, w)` for ",
        "Distributions.jl), or implement `HiddenMarkovModels.fit_duration_in_sequence!` for ",
        "this type.",
    )
end

#=
Whether `err`, thrown by `fit_in_sequence!(dists, i, x, w)`, means that `dist = dists[i]` has no
weighted fit. Only the outermost missing method is recognized, so that a `MethodError` raised
inside an existing fit still propagates. The Distributions.jl extension adds the "not
implemented" error of its generic `suffstats`.
=#
function is_missing_weighted_fit(err, dist)
    return err isa MethodError && err.f === StatsAPI.fit! && first(err.args) === dist
end

"""
    impute_censored_durations(dist, completed, censored; max_imputed_duration)

Return the weights `w[d]`, `d = 1, ..., L`, of the sojourn lengths once the right-censored
sojourns are completed under `dist`: each `censored[d]` is spread over `d' ≥ d` proportionally to
`p(d') / S(d)`, and added to `completed`.

The sum over `d ≤ d'` is accumulated in log space as `d'` grows, so each duration costs one
evaluation of `duration_logdensityof`. `L` is extended until the imputed mass beyond it is at
most `eps` times the total weight, which is the precision of the weights anyway, or until the
support of `dist` ends. The survival needed for this check is only evaluated at checkpoints whose
spacing doubles, i.e. O(log L) times, at the price of imputing at most twice as many durations as
needed. An `ArgumentError` is thrown if this needs more than `max_imputed_duration` durations,
rather than truncating silently.
"""
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
    #= `log_spread = log Σ_{d ≤ d'} censored[d] / S(d)`, so that the imputed weight of `d'` is
    `p(d') * exp(log_spread)`. =#
    log_spread = log_zero
    for (d, c) in enumerate(censored)
        if c > zero(c)
            log_surv = convert(R, duration_logsurvival(dist, d))
            check_not_nan(log_surv, dist, d, "duration_logsurvival")
            if is_log_zero(log_surv)
                #= The E-step gives no weight to an impossible censored sojourn, up to rounding
                where `S(d)` underflows. =#
                c <= negligible || throw(
                    ArgumentError(
                        "Censored sojourns of length $d have weight $c but zero survival " *
                        "under $dist. The duration counts must come from `forward_backward!` " *
                        "with the current duration distributions.",
                    ),
                )
            else
                log_spread = logaddexp_safe(log_spread, log(convert(R, c)) - log_surv)
            end
        end
        add_imputed_weight!(weights, d, dist, log_spread)
    end
    is_log_zero(log_spread) && return weights
    # Beyond the longest counted sojourn, the imputed weights follow `p(d') * exp(log_spread)`.
    log_negligible = log(negligible)
    L = length(weights)
    stride = 1
    while true
        checkpoint = min(L + stride, max(max_imputed_duration, L))
        while L < checkpoint
            L += 1
            push!(weights, zero(R))
            add_imputed_weight!(weights, L, dist, log_spread)
        end
        # The mass left beyond `L` is `S(L + 1) * exp(log_spread)`, zero after the support.
        log_surv = convert(R, duration_logsurvival(dist, L + 1))
        check_not_nan(log_surv, dist, L + 1, "duration_logsurvival")
        log_spread + log_surv <= log_negligible && break
        L >= max_imputed_duration && throw(
            ArgumentError(
                "The imputed tail of censored sojourns under $dist still holds a relative " *
                "mass above eps after $max_imputed_duration durations. Implement " *
                "`HiddenMarkovModels.fit_duration_in_sequence!` for this duration type, or " *
                "call `HiddenMarkovModels.fit_duration_by_imputation!` with a larger " *
                "`max_imputed_duration`.",
            ),
        )
        stride *= 2
    end
    # The last checkpoint may lie beyond the support, where the weights are zero.
    while length(weights) > length(completed) && iszero(last(weights))
        pop!(weights)
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
    isnan(x) && throw(ArgumentError("`$name` returned NaN for $dist at duration $d."))
    return nothing
end

"""
    fit_duration_by_imputation!(dists, i, completed, censored; max_imputed_duration)

Generic implementation of [`HiddenMarkovModels.fit_duration_in_sequence!`](@ref).

The censored sojourns are completed under the current distribution
`dists[i]` (see [`HiddenMarkovModels.impute_censored_durations`](@ref)), and `dists[i]` is then
refitted with the weighted [`HiddenMarkovModels.fit_in_sequence!`](@ref). Treating the
unobserved remainder of the censored sojourns as missing data, this is an exact EM step for `Q`,
so `Q` does not decrease (provided the weighted fit maximizes the weighted loglikelihood), and
iterating it converges to a maximizer of `Q` under the usual regularity conditions of EM.
See the formulas page of the documentation for the derivation.

Throws a [`HiddenMarkovModels.DurationFitError`](@ref) if the type of `dists[i]` has no weighted
fit. Errors raised by the weighted fit itself propagate unchanged.
"""
function fit_duration_by_imputation!(
    dists::AbstractVector,
    i::Integer,
    completed::AbstractVector,
    censored::AbstractVector;
    max_imputed_duration::Integer=DEFAULT_MAX_IMPUTED_DURATION,
)
    check_completed_sojourns(completed)
    weights = impute_censored_durations(dists[i], completed, censored; max_imputed_duration)
    # The distributions model `sojourn - 1`.
    shifted_durations = 0:(length(weights) - 1)
    dist = dists[i]
    try
        fit_in_sequence!(dists, i, shifted_durations, weights)
    catch err
        is_missing_weighted_fit(err, dist) && throw(DurationFitError(typeof(dist)))
        rethrow()
    end
    return nothing
end
