using Pkg

# Dev the sibling HMMBenchmark package from this checkout so both benchpkg
# revisions use the PR-head version of the suite-builder. This keeps the
# benchmark workload consistent across runs while varying only the
# HiddenMarkovModels code under test, and lets PRs that modify
# libs/HMMBenchmark exercise their own changes.
Pkg.develop(
    path=normpath(joinpath(@__DIR__, "..", "libs", "HMMBenchmark"));
    io=devnull,
)

include(joinpath(@__DIR__, "benchmarks.jl"))
