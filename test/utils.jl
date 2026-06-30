using DensityInterface
using Random: Random, AbstractRNG
# defined separately so we can import in `runtests.jl` so JET sees a concrete ControlledEmission subtype.
struct TestControlledEmissionDist <: ControlledEmission end

# `DensityKind` is inherited from the `ControlledEmission` abstract supertype.
function DensityInterface.logdensityof(::TestControlledEmissionDist, obs, control)
    return -(obs - control)^2
end
Random.rand(::AbstractRNG, ::TestControlledEmissionDist, control) = control
