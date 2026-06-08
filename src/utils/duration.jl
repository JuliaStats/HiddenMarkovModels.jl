"""
$(SIGNATURES)

Log-probability that a state with sojourn-time distribution `dist` is held for exactly `k`
consecutive timesteps.

`dist` is supplied by the user with support on `{0, 1, 2, ...}` and is interpreted as the law
of `(sojourn time - 1)`. The sojourn time itself lives on `{1, 2, 3, ...}`, so this returns
`logdensityof(dist, k - 1)`. Durations `k < 1` fall outside the user's support and yield
`-Inf` through `dist` (e.g. a `Distributions.Distribution` returns `-Inf` for a negative
argument).

See also [`rand_duration`](@ref).
"""
duration_logdensityof(dist, k::Integer) = logdensityof(dist, k - 1)

"""
$(SIGNATURES)

Sample a sojourn time on `{1, 2, 3, ...}` from the duration distribution `dist`.

`dist` is interpreted as the law of `(sojourn time - 1)` (see [`duration_logdensityof`](@ref)),
so this returns `rand(rng, dist) + 1`.
"""
rand_duration(rng::AbstractRNG, dist) = rand(rng, dist) + 1
rand_duration(dist) = rand_duration(default_rng(), dist)
