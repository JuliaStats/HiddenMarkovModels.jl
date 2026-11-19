using DensityInterface
using HiddenMarkovModels: duration_logdensityof
using Random: Random, AbstractRNG
# defined separately so we can import in `runtests.jl` so JET sees a concrete ControlledEmission subtype.
struct TestControlledEmissionDist <: ControlledEmission end

# `DensityKind` is inherited from the `ControlledEmission` abstract supertype.
function DensityInterface.logdensityof(::TestControlledEmissionDist, obs, control)
    return -(obs - control)^2
end
Random.rand(::AbstractRNG, ::TestControlledEmissionDist, control) = control

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
