```@meta
CollapsedDocStrings = true
```

# API reference

```@docs
HiddenMarkovModels
```

## Sequence formatting

Most algorithms below ingest the data with two positional arguments `obs_seq` (mandatory) and `control_seq` (optional), and a keyword argument `seq_ends` (optional).

- If the data consists of a single sequence, `obs_seq` and `control_seq` are the corresponding vectors of observations and controls, and you don't need to provide `seq_ends`.
- If the data consists of multiple sequences, `obs_seq` and `control_seq` are concatenations of several vectors, whose end indices are given by `seq_ends`. Starting from separate sequences `obs_seqs` and `control_seqs`, you can run the following snippet:

```julia
obs_seq = reduce(vcat, obs_seqs)
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(obs_seqs))
```

## Types

```@docs
AbstractHMM
HMM
AbstractHSMM
HSMM
```

## Interface

```@docs
initialization
transition_matrix
obs_distributions
duration_distributions
```

## Duration distributions

The per-state sojourn distributions used by [`HSMM`](@ref) live on the positive integers `{1, 2, 3, ...}` and form their own small type hierarchy. They are decoupled from `Distributions.jl` so an HSMM does not require it as a dependency.

```@docs
AbstractDurationDistribution
GeometricDuration
PoissonDuration
NegBinomialDuration
```

## Utils

```@docs
length
rand
eltype
seq_limits
```

## Inference

```@docs
logdensityof
joint_logdensityof
forward
viterbi
forward_backward
```

The same five entry points dispatch on [`AbstractHSMM`](@ref), routing to segment-based dynamic programming that accounts for sojourn-time distributions.

## Learning

```@docs
baum_welch
fit!
```

## In-place versions

### Forward

```@docs
HiddenMarkovModels.ForwardStorage
HiddenMarkovModels.initialize_forward
HiddenMarkovModels.forward!
HiddenMarkovModels.HSMMForwardStorage
HiddenMarkovModels.initialize_hsmm_forward
```

### Viterbi

```@docs
HiddenMarkovModels.ViterbiStorage
HiddenMarkovModels.initialize_viterbi
HiddenMarkovModels.viterbi!
HiddenMarkovModels.HSMMViterbiStorage
HiddenMarkovModels.initialize_hsmm_viterbi
```

### Forward-backward

```@docs
HiddenMarkovModels.ForwardBackwardStorage
HiddenMarkovModels.initialize_forward_backward
HiddenMarkovModels.forward_backward!
HiddenMarkovModels.HSMMForwardBackwardStorage
HiddenMarkovModels.initialize_hsmm_forward_backward
```

### Baum-Welch

```@docs
HiddenMarkovModels.baum_welch!
```

For an `AbstractHSMM`, `forward!`, `viterbi!`, `forward_backward!`, and `baum_welch!` dispatch on the corresponding HSMM storage types listed above.

## Miscellaneous

```@docs
HiddenMarkovModels.valid_hmm
HiddenMarkovModels.valid_hsmm
HiddenMarkovModels.rand_prob_vec
HiddenMarkovModels.rand_trans_mat
HiddenMarkovModels.fit_in_sequence!
```

## Internals

```@docs
HiddenMarkovModels.LightDiagNormal
HiddenMarkovModels.LightCategorical
HiddenMarkovModels.log_initialization
HiddenMarkovModels.log_transition_matrix
HiddenMarkovModels.mul_rows_cols!
HiddenMarkovModels.argmaxplus_transmul!
```

## Index

```@index
```
