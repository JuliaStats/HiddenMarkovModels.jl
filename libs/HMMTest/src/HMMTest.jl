module HMMTest

using BenchmarkTools: @ballocated
using HiddenMarkovModels
using HiddenMarkovModels: AbstractLatentStateModel, AbstractVectorOrNTuple
import HiddenMarkovModels as HMMs
using JET: @test_opt, @test_call
using Random: AbstractRNG
using Statistics: mean
using Test: @test, @testset, @test_broken

function test_identical_hmmbase end  # in extension

export transpose_hmm
export test_equal_hmms, test_coherent_algorithms
export test_identical_hmmbase
export test_allocations
export test_type_stability

include("utils.jl")
include("coherence.jl")
include("allocations.jl")
include("jet.jl")

end
