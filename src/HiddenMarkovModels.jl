"""
    HiddenMarkovModels

A Julia package for HMM modeling, simulation, inference and learning.

# Exports

$(EXPORTS)
"""
module HiddenMarkovModels

using ArgCheck: @argcheck
using Base: RefValue
using Base.Threads: @threads
using ChainRulesCore: ChainRulesCore, NoTangent, RuleConfig, rrule_via_ad
using DensityInterface: DensityInterface, DensityKind, HasDensity, NoDensity, logdensityof
using DocStringExtensions
using FillArrays: Fill
using LinearAlgebra: Transpose, axpy!, dot, ldiv!, lmul!, mul!, parent
using ProgressLogging: @withprogress, @logprogress
using Random: Random, AbstractRNG, default_rng
using SparseArrays: AbstractSparseArray, SparseMatrixCSC, nonzeros, nnz, nzrange, rowvals
using SpecialFunctions: trigamma
using StatsAPI: StatsAPI, fit, fit!
using StatsFuns: log2π, logaddexp, logsumexp, digamma, nbinomlogpdf, poislogpdf

export AbstractHMM, HMM
export AbstractHSMM, HSMM
export AbstractDurationDistribution, GeometricDuration, PoissonDuration, NegBinomialDuration
export initialization, transition_matrix, obs_distributions, duration_distributions
export log_initialization, log_transition_matrix
export fit!, logdensityof, joint_logdensityof
export viterbi, forward, forward_backward, baum_welch
export seq_limits

include("types/abstract_hmm.jl")
include("types/abstract_hsmm.jl")

include("utils/linalg.jl")
include("utils/valid.jl")
include("utils/valid_hsmm.jl")
include("utils/probvec_transmat.jl")
include("utils/fit.jl")
include("utils/lightdiagnormal.jl")
include("utils/lightcategorical.jl")
include("utils/limits.jl")

include("inference/predict.jl")
include("inference/forward.jl")
include("inference/viterbi.jl")
include("inference/forward_backward.jl")
include("inference/baum_welch.jl")
include("inference/logdensity.jl")
include("inference/chainrules.jl")

include("inference/forward_hsmm.jl")
include("inference/viterbi_hsmm.jl")
include("inference/forward_backward_hsmm.jl")
include("inference/baum_welch_hsmm.jl")

include("types/hmm.jl")
include("types/duration_distribution.jl")
include("types/hsmm.jl")

end
