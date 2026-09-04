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
result must be `0` for `k <= 1`.

A closed-form method based on `logccdf` is provided for every
`Distributions.DiscreteUnivariateDistribution` when Distributions.jl is loaded. Custom duration
distributions must implement this function themselves, for instance as

```julia
function HiddenMarkovModels.duration_logsurvival(dist::MyDuration, k::Integer)
    k <= 1 && return 0.0
    return log1p(-sum(exp(duration_logdensityof(dist, j)) for j in 1:(k - 1)))
end
```
"""
function duration_logsurvival end
