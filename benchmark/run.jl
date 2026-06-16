using BenchmarkTools

include("benchmarks.jl")

tag = isempty(ARGS) ? "results" : ARGS[1]
out = path -> joinpath(@__DIR__, path)

print_julia_setup(out("$tag-setup.txt"))

results = run(SUITE; verbose=true)

BenchmarkTools.save(out("$tag.json"), results)
parse_results(results; path=out("$tag.csv"))

println("\nWrote:")
println("  ", out("$tag.json"))
println("  ", out("$tag.csv"))
println("  ", out("$tag-setup.txt"))
