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
$(SIGNATURES)

Modify the `i`-th element of `dists` to maximize

    Q(θ) = Σ_d completed[d] * log p(d; θ) + Σ_d censored[d] * log S(d; θ)

where `p` and `S` are given by [`duration_logdensityof`](@ref) and
[`duration_logsurvival`](@ref), and `completed[d]` and `censored[d]` are the expected numbers of
completed and right-censored sojourns lasting `d` steps. A method must not decrease `Q`, and can
assume `sum(completed) > 0`.

Default behavior: impute the remainder of the censored sojourns under the current `dists[i]`
(an EM step for `Q`, see the formulas page), then call
[`HiddenMarkovModels.fit_in_sequence!`](@ref) on the durations `0:(L-1)` with the resulting
weights.

Override for Distributions.jl (in the package extension): a closed form for `Geometric`.
"""
function fit_duration_in_sequence!(
    dists::AbstractVector, i::Integer, completed::AbstractVector, censored::AbstractVector
)
    @argcheck sum(completed) > 0 "Cannot fit the duration of state $i because none of its \
    stays ends inside a sequence. Try skipping this state and keeping its current duration \
    distribution."
    fit_duration!(dists, i, dists[i], completed, censored)
    return nothing
end

# Dispatches on `dists[i]`, so that the closed forms also apply when `dists` mixes families.
function fit_duration!(
    dists::AbstractVector,
    i::Integer,
    dist,
    completed::AbstractVector,
    censored::AbstractVector,
)
    weights = impute_censored_durations(dist, completed, censored)
    fit_in_sequence!(dists, i, 0:(length(weights) - 1), weights)
    return nothing
end

#=
Return the weights `w[d]`, `d = 1, ..., L`, of the sojourn lengths once the right-censored
sojourns are completed under `dist`: each `censored[d]` is spread over `d' ≥ d` proportionally to
`p(d') / S(d)`, and added to `completed`. `L` grows until the imputed mass beyond it is below
`eps` times the total weight, or the support of `dist` ends.
=#
function impute_censored_durations(
    dist, completed::AbstractVector, censored::AbstractVector
)
    R = float(promote_type(eltype(completed), eltype(censored)))
    weights = collect(R, completed)
    total = sum(weights) + sum(censored)
    # Weights below `eps(R) * total` are lost to rounding (in log space, which cannot underflow).
    log_negligible = log(eps(R)) + log(total)
    # `log_spread = log Σ_{d ≤ d'} censored[d] / S(d)`, so that `d'` gets `p(d') * exp(log_spread)`.
    log_spread = convert(R, -Inf)
    for d in eachindex(completed, censored)
        c = censored[d]
        if c > zero(c)
            log_surv = convert(R, duration_logsurvival(dist, d))
            if is_log_zero(log_surv)
                @argcheck log(convert(R, c)) <= log_negligible "A state is still active \
                after $d steps, but its duration distribution $dist says it cannot last that \
                long. Try checking that `duration_logsurvival` agrees with \
                `duration_logdensityof` for this distribution."
            else
                log_spread = logaddexp_safe(log_spread, log(convert(R, c)) - log_surv)
            end
        end
        weights[d] += exp(convert(R, duration_logdensityof(dist, d)) + log_spread)
    end
    L = length(weights)
    # The mass left beyond `L` is `S(L + 1) * exp(log_spread)`. A NaN also stops the tail.
    while log_spread + convert(R, duration_logsurvival(dist, L + 1)) > log_negligible
        @argcheck L < 2^20 "The duration distribution $dist puts too much weight on stays \
        longer than $(2^20) steps, which is too many to compute. Try a duration distribution \
        with a smaller mean."
        L += 1
        push!(weights, exp(convert(R, duration_logdensityof(dist, L)) + log_spread))
    end
    return weights
end
