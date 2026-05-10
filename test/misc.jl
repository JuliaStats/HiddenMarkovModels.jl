using HiddenMarkovModels
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using Distributions
using Logging: with_logger
using ProgressLogging: ProgressLevel
using Random: Xoshiro
using Test
using Test: TestLogger

@testset "Allow NaN density" begin
    init = rand_prob_vec(2)
    trans = rand_trans_mat(2)
    dists = [Normal(i, Inf) for i in 1:2]
    hmm = HMM(init, trans, dists)
    obs_seq = rand(5)
    @test isnan(logdensityof(hmm, obs_seq))
end

@testset "Baum-Welch progress kwarg" begin
    rng = Xoshiro(0)
    init = [0.4, 0.6]
    trans = [0.7 0.3; 0.3 0.7]
    dists = [Normal(-1.0), Normal(1.0)]
    hmm = HMM(init, trans, dists)
    _, obs_seq = rand(rng, hmm, 200)

    init_guess = [0.5, 0.5]
    trans_guess = [0.6 0.4; 0.4 0.6]
    dists_guess = [Normal(-0.8), Normal(0.8)]
    hmm_guess = HMM(init_guess, trans_guess, dists_guess)

    # progress=false: no progress records emitted
    silent = TestLogger(; min_level=ProgressLevel)
    with_logger(silent) do
        baum_welch(hmm_guess, obs_seq; max_iterations=5, progress=false)
    end
    @test !any(r -> haskey(r.kwargs, :progress), silent.logs)

    # progress=true: progress records emitted, one per iteration plus the
    # @withprogress begin/end markers
    chatty = TestLogger(; min_level=ProgressLevel)
    _, logL = with_logger(chatty) do
        baum_welch(hmm_guess, obs_seq; max_iterations=5, progress=true)
    end
    progress_logs = filter(r -> haskey(r.kwargs, :progress), chatty.logs)
    @test length(progress_logs) >= length(logL)
    fractions = [r.kwargs[:progress] for r in progress_logs]
    @test any(f -> f isa Real && 0 < f <= 1, fractions)

    # results match between the two paths
    silent2 = TestLogger(; min_level=ProgressLevel)
    hmm_a, logL_a = with_logger(silent2) do
        baum_welch(hmm_guess, obs_seq; max_iterations=5, progress=false)
    end
    chatty2 = TestLogger(; min_level=ProgressLevel)
    hmm_b, logL_b = with_logger(chatty2) do
        baum_welch(hmm_guess, obs_seq; max_iterations=5, progress=true)
    end
    @test logL_a == logL_b
    @test transition_matrix(hmm_a) == transition_matrix(hmm_b)
end
