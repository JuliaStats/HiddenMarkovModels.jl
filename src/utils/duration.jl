# Stop once the tail converges numerically, with a hard limit for pathological distributions.
const MAX_QUIET_TERMS = 64
const MAX_TAIL_TERMS = 100_000

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
$(SIGNATURES)

Return `log ℙ(sojourn time >= k)` for the duration distribution `dist`.

This is the right-censoring term for the final segment of a sequence. The generic method sums the
tail until it converges numerically; duration types may provide a closed form instead.
"""
function duration_logsurvival(dist, k::Integer)
    # Every sojourn lasts at least one timestep.
    k <= one(k) && return zero(duration_logdensityof(dist, one(k)))
    log_tail = oftype(duration_logdensityof(dist, one(k)), -Inf)
    quiet = 0
    for d in k:(k + MAX_TAIL_TERMS)
        new_tail = logaddexp(log_tail, duration_logdensityof(dist, d))
        if new_tail == log_tail
            quiet += 1
            quiet >= MAX_QUIET_TERMS && break
        else
            quiet = 0
        end
        log_tail = new_tail
    end
    return log_tail
end
