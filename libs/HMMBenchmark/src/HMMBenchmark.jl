module HMMBenchmark

using Base.Threads: Threads
using InteractiveUtils: InteractiveUtils
using BenchmarkTools: @benchmarkable, BenchmarkGroup
using CSV: CSV
using DataFrames: DataFrame
using Distributions: Normal, MvNormal, Poisson
using HiddenMarkovModels
using HiddenMarkovModels:
    LightDiagNormal,
    initialize_baum_welch,
    rand_prob_vec,
    rand_trans_mat,
    initialize_viterbi,
    viterbi!,
    initialize_forward,
    forward!,
    initialize_forward_backward,
    forward_backward!,
    baum_welch!
using LinearAlgebra: BLAS, Diagonal, SymTridiagonal
using Pkg: Pkg
using Random: AbstractRNG
using SparseArrays: spdiagm
using Statistics: mean, median, std, quantile

export Implementation, Instance, Params, HiddenMarkovModelsImplem
export build_params, build_data, build_model, build_benchmarkables
export define_suite, parse_results, read_results, print_julia_setup

abstract type Implementation end

include("instance.jl")
include("params.jl")
include("hiddenmarkovmodels.jl")
include("suite.jl")
include("setup.jl")

end
