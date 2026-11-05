using HiddenMarkovModels
using HiddenMarkovModels: forward_backward!, initialize_forward_backward, valid_hsmm
using Distributions: Categorical, Geometric, Normal, Poisson, mean, std, succprob
using JET: @test_call, @test_opt
using SparseArrays: SparseMatrixCSC, nnz, sparse
using StableRNGs: StableRNG
using Test

function sample_sequences(rng, hsmm, lengths)
    obs_seq = reduce(vcat, [rand(rng, hsmm, T).obs_seq for T in lengths])
    return obs_seq, cumsum(lengths)
end

@testset "Recovery" begin
    init = [0.5, 0.3, 0.2]
    trans = [0.0 0.7 0.3; 0.4 0.0 0.6; 0.5 0.5 0.0]
    dists = [Normal(-3.0), Normal(0.0), Normal(3.0)]
    dur_dists = [Poisson(4.0), Poisson(1.0), Poisson(7.0)]
    obs_seq, seq_ends = sample_sequences(
        StableRNG(506), HSMM(init, trans, dists, dur_dists), fill(200, 50)
    )
    hsmm_guess = HSMM(
        fill(1 / 3, 3),
        [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0],
        [Normal(-2.0), Normal(0.5), Normal(2.0)],
        fill(Poisson(2.0), 3),
    )
    for max_duration in (10, 30)
        hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends, max_duration)
        @test all(>=(0), diff(logL_evolution))
        @test valid_hsmm(hsmm_est)
        max_duration == 10 && continue
        @test all(isapprox.(hsmm_est.init, init; atol=0.15))
        @test all(isapprox.(hsmm_est.trans, trans; atol=0.05))
        @test all(isapprox.(mean.(hsmm_est.dists), mean.(dists); atol=0.1))
        @test all(isapprox.(std.(hsmm_est.dists), std.(dists); atol=0.05))
        @test all(isapprox.(mean.(hsmm_est.dur_dists), mean.(dur_dists); rtol=0.08))
    end
    @test hsmm_guess.dur_dists == fill(Poisson(2.0), 3)
end

@testset "Short sequences" begin
    # Sequences of 2.5 mean sojourns, where leaving out the censored sojourns biases the durations.
    dists = [Normal(-2.0), Normal(2.0)]
    dur_dists = [Poisson(7.0), Poisson(4.0)]
    obs_seq, seq_ends = sample_sequences(
        StableRNG(513),
        HSMM([0.5, 0.5], [0.0 1.0; 1.0 0.0], dists, dur_dists),
        fill(20, 200),
    )
    hsmm_guess = HSMM(
        [0.6, 0.4],
        [0.0 1.0; 1.0 0.0],
        [Normal(-1.0, 1.5), Normal(1.0, 1.5)],
        [Poisson(3.0), Poisson(10.0)],
    )
    hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends)
    @test all(>=(0), diff(logL_evolution))
    @test all(isapprox.(mean.(hsmm_est.dur_dists), mean.(dur_dists); rtol=0.05))
    @test all(isapprox.(mean.(hsmm_est.dists), mean.(dists); atol=0.05))
end

@testset "HMM equivalence" begin
    # With geometric durations, both models and both algorithms coincide.
    hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
    hmm = HMM([0.5, 0.3, 0.2], hmm_trans, [Normal(0.0), Normal(5.0), Normal(10.0)])
    hsmm_from_hmm(hmm) = HSMM(
        hmm.init,
        [i == j ? 0.0 : hmm.trans[i, j] / (1 - hmm.trans[i, i]) for i in 1:3, j in 1:3],
        hmm.dists,
        [Geometric(1 - hmm.trans[i, i]) for i in 1:3],
    )
    # Short sequences make the censored sojourns matter.
    for lengths in ([80, 120], fill(6, 30))
        obs_seq, seq_ends = sample_sequences(StableRNG(length(lengths)), hmm, lengths)
        hmm_est, logL_hmm = baum_welch(hmm, obs_seq; seq_ends, max_iterations=10)
        hsmm_est, logL_hsmm = baum_welch(
            hsmm_from_hmm(hmm), obs_seq; seq_ends, max_iterations=10
        )
        hsmm_ref = hsmm_from_hmm(hmm_est)
        @test logL_hsmm ≈ logL_hmm
        @test hsmm_est.init ≈ hsmm_ref.init
        @test hsmm_est.trans ≈ hsmm_ref.trans
        @test succprob.(hsmm_est.dur_dists) ≈ succprob.(hsmm_ref.dur_dists)
        @test mean.(hsmm_est.dists) ≈ mean.(hsmm_ref.dists)
        @test std.(hsmm_est.dists) ≈ std.(hsmm_ref.dists)
    end
end

@testset "Multiple sequences" begin
    hsmm = HSMM(
        [0.6, 0.4],
        [0.0 1.0; 1.0 0.0],
        [Normal(0.0), Normal(3.0)],
        [Geometric(0.3), Geometric(0.5)],
    )
    obs_seq, seq_ends = sample_sequences(StableRNG(507), hsmm, [30, 1, 45, 20])
    # Vectors of sequence ends go through the threaded loops, tuples do not.
    hsmm_vec, logL_vec = baum_welch(hsmm, obs_seq; seq_ends, max_iterations=10)
    hsmm_tup, logL_tup = baum_welch(
        hsmm, obs_seq; seq_ends=Tuple(seq_ends), max_iterations=10
    )
    @test logL_vec ≈ logL_tup
    @test hsmm_vec.trans ≈ hsmm_tup.trans
    @test succprob.(hsmm_vec.dur_dists) ≈ succprob.(hsmm_tup.dur_dists)
    # Empty sequences change nothing.
    for seq_ends_empty in ([0, 30, 30, 31, 76, 96], (30, 31, 31, 76, 96, 96))
        hsmm_empty, logL_empty = baum_welch(
            hsmm, obs_seq; seq_ends=seq_ends_empty, max_iterations=10, max_duration=45
        )
        @test logL_empty ≈ logL_vec
        @test hsmm_empty.init ≈ hsmm_vec.init
        @test hsmm_empty.trans ≈ hsmm_vec.trans
        @test succprob.(hsmm_empty.dur_dists) ≈ succprob.(hsmm_vec.dur_dists)
    end
end

@testset "Sparse" begin
    trans = sparse([0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0])
    dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
    durs = fill(Geometric(0.3), 3)
    obs_seq, seq_ends = sample_sequences(
        StableRNG(509), HSMM([0.5, 0.3, 0.2], trans, dists, durs), fill(100, 3)
    )
    hsmm_guess = HSMM(
        [0.5, 0.3, 0.2],
        sparse([0.0 0.3 0.7; 1.0 0.0 0.0; 1.0 0.0 0.0]),
        [Normal(0.5), Normal(2.5), Normal(5.0)],
        fill(Geometric(0.5), 3),
    )
    hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends)
    hsmm_dense, logL_dense = baum_welch(
        HSMM(
            hsmm_guess.init,
            Matrix(hsmm_guess.trans),
            hsmm_guess.dists,
            hsmm_guess.dur_dists,
        ),
        obs_seq;
        seq_ends,
    )
    @test all(>=(0), diff(logL_evolution))
    @test logL_evolution ≈ logL_dense
    @test hsmm_est.trans isa SparseMatrixCSC
    @test nnz(hsmm_est.trans) == nnz(hsmm_est.logtrans) == nnz(trans)
    @test hsmm_est.trans ≈ hsmm_dense.trans
    @test valid_hsmm(hsmm_est)
end

@testset "State never left" begin
    #= State 1 only emits 1 and state 2 only emits 2, and every sequence ends in state 2, so
    all the sojourns of state 2 are censored. =#
    rng = StableRNG(522)
    obs_seq = Int[]
    seq_ends = Int[]
    for _ in 1:20
        append!(obs_seq, fill(1, rand(rng, 2:8)), fill(2, rand(rng, 1:6)))
        push!(seq_ends, length(obs_seq))
    end
    dists = [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])]
    for dur_dists in ([Poisson(2.0), Poisson(3.0)], [Geometric(0.5), Geometric(0.3)])
        hsmm_guess = HSMM([0.5, 0.5], [0.0 1.0; 1.0 0.0], dists, dur_dists)
        hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends)
        @test all(>=(0), diff(logL_evolution))
        @test hsmm_est.init == [1.0, 0.0]
        @test hsmm_est.trans == [0.0 1.0; 1.0 0.0]
        @test hsmm_est.dur_dists[1] != dur_dists[1]
        @test hsmm_est.dur_dists[2] == dur_dists[2]
    end
end

@testset "Mixed precision" begin
    # Float32 observations and Float64 durations promote the storage to Float64.
    hsmm = HSMM(
        Float32[0.6, 0.4],
        Float32[0.0 1.0; 1.0 0.0],
        [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
        [Geometric(0.4), Geometric(0.6)],
    )
    obs_seq = Float32.(rand(StableRNG(519), hsmm, 200).obs_seq)
    hsmm_est, logL_evolution = baum_welch(hsmm, obs_seq)
    @test logL_evolution isa Vector{Float64}
    @test all(>=(0), diff(logL_evolution))
    @test hsmm_est.trans isa Matrix{Float32}
    @test hsmm_est.dur_dists isa Vector{Geometric{Float64}}
end

@testset "Type stability and errors" begin
    hsmm = HSMM(
        [0.5, 0.3, 0.2],
        [0.0 0.7 0.3; 0.4 0.0 0.6; 0.5 0.5 0.0],
        [Normal(-1.0), Normal(0.0), Normal(1.0)],
        [Poisson(2.0), Poisson(3.0), Poisson(1.0)],
    )
    obs_seq = rand(StableRNG(79), hsmm, 20).obs_seq
    control_seq = fill(nothing, 20)
    seq_ends = [5, 20]
    storage = initialize_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    @test_opt target_modules = (HiddenMarkovModels,) fit!(
        deepcopy(hsmm), storage, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) baum_welch(
        hsmm, obs_seq, control_seq; seq_ends, max_duration=5
    )
    @test_call target_modules = (HiddenMarkovModels,) baum_welch(hsmm, obs_seq; seq_ends)
    # Without duration counts, `fit!` cannot fit the durations.
    storage = initialize_forward_backward(hsmm, obs_seq, control_seq; seq_ends)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
end
