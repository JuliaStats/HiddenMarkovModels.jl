# Benchmarks

Tracks performance of `HiddenMarkovModels.jl`'s core algorithms (`forward`,
`viterbi`, `forward_backward`, `baum_welch`) across a small fixed instance grid
defined in [benchmarks.jl](benchmarks.jl).

The CI workflow ([.github/workflows/benchmark.yml](../.github/workflows/benchmark.yml))
runs on any PR tagged `run benchmark` and uses
[AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl) to
compare the PR head against the default branch, posting the results as a PR
comment.

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

## Regression workflow via AirspeedVelocity

CI uses the [`MilesCranmer/AirspeedVelocity.jl@action-v1`](https://github.com/MilesCranmer/AirspeedVelocity.jl)
composite action, which runs `benchpkg` against the default branch and the PR
head and posts a `benchpkgtable` comparison as a PR comment.

The workflow invokes [`benchmark/ci_benchmarks.jl`](ci_benchmarks.jl) as the
`script:` input rather than the default `benchmarks.jl`. That wrapper
`Pkg.develop`s [`../libs/HMMBenchmark`](../libs/HMMBenchmark) from the
checked-out PR head before including `benchmarks.jl`, which means **both
benchmarked revisions use the PR-head version of the suite-builder**. PRs
that modify `libs/HMMBenchmark` exercise their own changes, and the workload
stays identical across the baseline and target runs so the diff attributes
cleanly to `HiddenMarkovModels` itself.

To reproduce locally with `benchpkg`:

```bash
julia -e 'using Pkg; Pkg.add(name="AirspeedVelocity", version="0.6"); Pkg.build("AirspeedVelocity")'
export PATH="$HOME/.julia/bin:$PATH"

mkdir -p results
benchpkg HiddenMarkovModels \
  --rev=main,HEAD \
  --script=benchmark/ci_benchmarks.jl \
  --add=StableRNGs \
  --output-dir=results/ \
  --tune

benchpkgtable HiddenMarkovModels \
  --rev=main,HEAD \
  --input-dir=results/ \
  --mode=time --ratio
```

## Manual `judge` workflow

If you'd rather drive the comparison directly with `BenchmarkTools.judge`,
[compare.jl](compare.jl) takes two JSON files produced by `run.jl`:

```bash
git switch main
julia --project=benchmark benchmark/run.jl main

git switch your-branch
julia --project=benchmark benchmark/run.jl branch

julia --project=benchmark benchmark/compare.jl benchmark/main.json benchmark/branch.json
```

`compare.jl` runs `BenchmarkTools.judge(minimum(branch), minimum(main))` and
prints regressions / improvements per `(instance, algorithm)`.

## Customizing the suite

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
