# Benchmarks

Tracks performance of `HiddenMarkovModels.jl`'s core algorithms (`forward`,
`viterbi`, `forward_backward`, `baum_welch`) across a small fixed instance grid
defined in [benchmarks.jl](benchmarks.jl).

## One-shot run

```bash
julia --project=benchmark benchmark/run.jl
```

Writes three files in `benchmark/`:

- `results.json` — raw `BenchmarkTools` group, readable with
  `BenchmarkTools.load`. Use this for `judge` / regression analysis.
- `results.csv` — one row per `(implementation, algorithm, instance)` with
  aggregated timings (min/median/max/mean/std/quantiles).
- `results-setup.txt` — `versioninfo()`, thread counts, and `Pkg.status()`
  snapshot for the run.

An optional positional argument tags the output filenames:

```bash
julia --project=benchmark benchmark/run.jl main      # writes main.json, main.csv, ...
```

## Regression workflow

To measure the perf impact of a branch against `main`:

```bash
git switch main
julia --project=benchmark benchmark/run.jl main

git switch your-branch
julia --project=benchmark benchmark/run.jl branch

julia --project=benchmark benchmark/compare.jl benchmark/main.json benchmark/branch.json
```

`compare.jl` runs `BenchmarkTools.judge(minimum(branch), minimum(main))` and
prints regressions / improvements per `(instance, algorithm)`. Run both
measurements on the same machine, with the same thread count, and ideally with
nothing else competing for CPU.

```julia
SUITE = define_suite(rng; instances, algos)
# or, with explicit budget per benchmark:
for instance in instances
    params = build_params(rng, instance)
    data = build_data(rng, instance)
    benchs = build_benchmarkables(
        HiddenMarkovModelsImplem(), instance, params, data, algos;
        samples=200, seconds=20,
    )
    # ... assign into SUITE
end
```
