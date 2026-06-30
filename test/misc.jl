using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: rand_prob_vec, rand_trans_mat
using DensityInterface
using Distributions
using Logging: with_logger
using ProgressLogging: ProgressLevel
using Random: Random, AbstractRNG, Xoshiro
using Test
using Test: TestLogger

function thrown_error(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

@testset "Allow NaN density" begin
    init = rand_prob_vec(2)
    trans = rand_trans_mat(2)
    dists = [Normal(i, Inf) for i in 1:2]
    hmm = HMM(init, trans, dists)
    obs_seq = rand(5)
    @test isnan(logdensityof(hmm, obs_seq))
end

@testset "ControlledEmissionHMM accessors and control errors" begin
    init = [0.4, 0.6]
    trans = [0.7 0.3; 0.2 0.8]
    dists = [TestControlledEmissionDist(), TestControlledEmissionDist()]
    hmm = ControlledEmissionHMM(init, trans, dists)

    @test initialization(hmm) === init
    @test HMMs.log_initialization(hmm) === hmm.loginit
    @test transition_matrix(hmm) === trans
    @test HMMs.log_transition_matrix(hmm) === hmm.logtrans
    @test transition_matrix(hmm, 1.0) === trans
    @test transition_matrix(hmm, nothing) === trans
    @test HMMs.log_transition_matrix(hmm, 1.0) === hmm.logtrans
    @test HMMs.log_transition_matrix(hmm, nothing) === hmm.logtrans

    controlled_dists = obs_distributions(hmm, 2.0)
    @test controlled_dists isa ControlBoundEmissionVector
    @test controlled_dists[1] isa ControlBoundEmission
    @test logdensityof(controlled_dists[1], 3.0) == -1.0

    nothing_dists = obs_distributions(hmm, nothing)
    @test nothing_dists isa ControlBoundEmissionVector
    @test nothing_dists[1] isa ControlBoundEmission

    err = thrown_error() do
        rand(hmm, 3)
    end
    @test err isa MethodError
    @test occursin("requires a control sequence", sprint(showerror, err))
    @test occursin("rand(hmm, control_seq)", sprint(showerror, err))

    err = thrown_error() do
        rand(Random.default_rng(), hmm, 3)
    end
    @test err isa MethodError
    @test occursin("requires a control sequence", sprint(showerror, err))
    @test occursin("rand(hmm, control_seq)", sprint(showerror, err))

    obs_seq = [1.0, 2.0, 3.0]
    control_seq = [1.0, 1.5, 2.0]
    seq_ends = (3,)
    fb_storage = HMMs.initialize_forward_backward(hmm, obs_seq, control_seq; seq_ends)
    err = thrown_error() do
        fit!(hmm, fb_storage, obs_seq; seq_ends)
    end
    @test err isa MethodError
    @test occursin("requires `control_seq`", sprint(showerror, err))
    @test occursin("fit!(hmm, fb_storage, obs_seq, control_seq", sprint(showerror, err))
end

@testset "ControlBoundEmission DensityKind and ControlBoundEmissionVector eltype" begin
    dist = TestControlledEmissionDist()
    # `DensityKind` is provided by the `ControlledEmission` abstract supertype.
    @test DensityInterface.DensityKind(dist) === DensityInterface.HasDensity()

    ce = ControlBoundEmission(dist, 1.5)
    @test DensityInterface.DensityKind(ce) === DensityInterface.HasDensity()

    dists = [TestControlledEmissionDist(), TestControlledEmissionDist()]
    ces = ControlBoundEmissionVector(dists, 2.5)
    @test eltype(ces) === ControlBoundEmission{TestControlledEmissionDist,Float64}
    @test eltype(typeof(ces)) === ControlBoundEmission{TestControlledEmissionDist,Float64}
end

@testset "ControlledEmissionHMM constructor rejects invalid inputs" begin
    dists2 = [TestControlledEmissionDist(), TestControlledEmissionDist()]
    dists3 = [
        TestControlledEmissionDist(),
        TestControlledEmissionDist(),
        TestControlledEmissionDist(),
    ]
    valid_init = [0.4, 0.6]
    valid_trans = [0.7 0.3; 0.2 0.8]

    # length mismatch between dists and init/trans dimensions
    @test_throws ArgumentError ControlledEmissionHMM(valid_init, valid_trans, dists3)

    # invalid initial probability vector (does not sum to 1)
    @test_throws ArgumentError ControlledEmissionHMM([0.3, 0.6], valid_trans, dists2)

    # invalid transition matrix (rows do not sum to 1)
    bad_trans = [0.5 0.3; 0.2 0.8]
    @test_throws ArgumentError ControlledEmissionHMM(valid_init, bad_trans, dists2)
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
