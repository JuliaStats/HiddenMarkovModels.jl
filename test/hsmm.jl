using DensityInterface: DensityKind, HasDensity
using HiddenMarkovModels
using HiddenMarkovModels:
    AbstractHSMM,
    duration_distributions,
    elementwise_log,
    log_initialization,
    log_transition_matrix,
    valid_hsmm
using Distributions: Geometric, Normal
using StableRNGs: StableRNG
using Test

@testset "HSMM type" begin
    init = [0.6, 0.4]
    trans = [0.0 1.0; 1.0 0.0]
    dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
    dur_dists = [Geometric(0.4), Geometric(0.6)]

    @testset "Construction and accessors" begin
        hsmm = HSMM(init, trans, dists, dur_dists)
        @test hsmm isa AbstractHSMM
        @test !(hsmm isa AbstractHMM)
        @test length(hsmm) == 2
        @test initialization(hsmm) == init
        @test transition_matrix(hsmm) == trans
        @test obs_distributions(hsmm) === dists
        @test duration_distributions(hsmm) === dur_dists
        @test valid_hsmm(hsmm)
        @test log_initialization(hsmm) == elementwise_log(init)
        @test log_transition_matrix(hsmm) == elementwise_log(trans)
        @test DensityKind(hsmm) == HasDensity()
    end

    @testset "Constructor rejects self-transitions (no silent zeroing)" begin
        bad_trans = [0.1 0.9; 1.0 0.0]
        @test_throws ArgumentError HSMM(init, bad_trans, dists, dur_dists)
    end

    @testset "Constructor rejects dimension mismatch" begin
        @test_throws ArgumentError HSMM(init, trans, dists, dur_dists[1:1])
        @test_throws ArgumentError HSMM([1.0], trans, dists, dur_dists)
    end
end

@testset "HSMM sampling" begin
    init = [1.0, 0.0, 0.0]
    trans = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0]
    dists = [Normal(0.0), Normal(5.0), Normal(10.0)]

    @testset "Return shape and bounds" begin
        dur_dists = [Geometric(0.3) for _ in 1:3]
        hsmm = HSMM(init, trans, dists, dur_dists)
        T = 200
        (; state_seq, obs_seq, duration_seq) = rand(StableRNG(7), hsmm, T)
        @test length(state_seq) == T
        @test length(obs_seq) == T
        @test length(duration_seq) == T
        @test all(s in 1:3 for s in state_seq)
        @test all(d >= 1 for d in duration_seq)
        @test state_seq[1] == 1  # init forces state 1
        @test eltype(obs_seq) == Float64
    end

    @testset "Unit sojourns forbid consecutive repeats" begin
        # Geometric(1.0) => the (sojourn-1) law is a point mass at 0, so every sojourn has length
        # 1; with a zero-diagonal transition matrix no two consecutive timesteps share a state.
        dur_dists = [Geometric(1.0) for _ in 1:3]
        hsmm = HSMM(init, trans, dists, dur_dists)
        (; state_seq, duration_seq) = rand(StableRNG(11), hsmm, 100)
        @test all(duration_seq .== 1)
        @test all(state_seq[t] != state_seq[t + 1] for t in 1:99)
    end
end

@testset "AbstractHMM <: AbstractHSMM" begin
    init = [0.5, 0.5]
    trans = [0.7 0.3; 0.2 0.8]
    dists = [Normal(0.0), Normal(5.0)]
    hmm = HMM(init, trans, dists)

    @testset "Subtype relationship" begin
        @test HMM <: AbstractHMM
        @test AbstractHMM <: AbstractHSMM
        @test hmm isa AbstractHSMM
    end

    @testset "HMM does not gain a duration_distributions method" begin
        # An HMM is structurally an AbstractHSMM for code reuse, but it deliberately has no
        # duration distributions: regular HMM usage is unaffected by the HSMM machinery.
        @test !hasmethod(duration_distributions, Tuple{typeof(hmm)})
    end

    @testset "HMM rand keeps 2-tuple shape (backward compatible)" begin
        out = rand(StableRNG(5), hmm, 50)
        @test propertynames(out) == (:state_seq, :obs_seq)
        @test !hasproperty(out, :duration_seq)
        @test length(out.state_seq) == 50
        @test length(out.obs_seq) == 50
    end
end
