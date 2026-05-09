# HMMComparison

To re-run the experiments from the JOSS paper, clone the repository:

```bash
git clone https://github.com/JuliaStats/HiddenMarkovModels.jl
cd HiddenMarkovModels.jl
```

Start a single-threaded Julia REPL in the comparison environment:

```bash
julia -t 1 --project=libs/HMMComparison
```

Then run the following files:

```julia
include("libs/HMMComparison/experiments/measurements.jl")
include("libs/HMMComparison/experiments/plots.jl")
```

The results and plots will appear in `libs/HMMComparison/experiments/results/`.
