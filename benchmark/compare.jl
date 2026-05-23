using BenchmarkTools
using Printf

const USAGE = """
Usage: julia --project=benchmark benchmark/compare.jl <baseline.json> <target.json>

Loads two BenchmarkTools result files (produced by benchmark/run.jl) and prints
a judge() comparison using the minimum-time estimator.
"""

if length(ARGS) != 2
    print(stderr, USAGE)
    exit(1)
end

baseline_path, target_path = ARGS
for p in (baseline_path, target_path)
    isfile(p) || error("Result file not found: $p")
end

baseline = BenchmarkTools.load(baseline_path)[1]
target = BenchmarkTools.load(target_path)[1]

j = judge(minimum(target), minimum(baseline))

@printf(
    "Comparison (estimator = minimum)\n  baseline: %s\n  target:   %s\n\n",
    baseline_path,
    target_path
)
show(stdout, MIME"text/plain"(), j)
println()

regs = BenchmarkTools.leaves(BenchmarkTools.regressions(j))
imps = BenchmarkTools.leaves(BenchmarkTools.improvements(j))
@printf("\n%d regression(s), %d improvement(s)\n", length(regs), length(imps))
