using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using HiddenMarkovModels
using HiddenMarkovModels:
    AbstractLatentStateModel,
    AbstractHSMM,
    duration_distributions,
    duration_logdensityof,
    duration_logsurvival,
    duration_logdensity_type,
    elementwise_log,
    log_initialization,
    log_transition_matrix,
    valid_hsmm
using Distributions: Geometric, Normal
using Random: AbstractRNG
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

@testset "duration_logdensity_type" begin
    init = [0.6, 0.4]
    trans = [0.0 1.0; 1.0 0.0]
    dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]

    @testset "HSMM returns the duration logdensity type" begin
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        @test duration_logdensity_type(hsmm, nothing) === Float64

        hsmm32 = HSMM(init, trans, dists, [Geometric(0.4f0), Geometric(0.6f0)])
        @test duration_logdensity_type(hsmm32, nothing) === Float32
    end

    @testset "Feeds into eltype promotion" begin
        #=
        Everything is Float32 except the duration logdensities, which must drive the
        promotion up to Float64.
        =#
        init32 = Float32[0.6, 0.4]
        trans32 = Float32[0.0 1.0; 1.0 0.0]
        dists32 = [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)]
        hsmm = HSMM(init32, trans32, dists32, [Geometric(0.4), Geometric(0.6)])
        @test eltype(hsmm, 0.0f0, nothing) === Float64
    end

    @testset "HMM eltype ignores duration machinery" begin
        hmm = HMM([0.5, 0.5], [0.7 0.3; 0.2 0.8], dists)
        @test !hasmethod(duration_logdensity_type, Tuple{typeof(hmm),Nothing})
        @test eltype(hmm, 0.0, nothing) === Float64
        hmm32 = HMM(
            Float32[0.5, 0.5],
            Float32[0.7 0.3; 0.2 0.8],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
        )
        @test eltype(hmm32, 0.0f0, nothing) === Float32
    end
end

@testset "Type hierarchy: AbstractHMM and AbstractHSMM are siblings" begin
    init = [0.5, 0.5]
    trans = [0.7 0.3; 0.2 0.8]
    dists = [Normal(0.0), Normal(5.0)]
    hmm = HMM(init, trans, dists)

    @testset "Sibling relationship under AbstractLatentStateModel" begin
        @test HMM <: AbstractHMM
        @test AbstractHMM <: AbstractLatentStateModel
        @test AbstractHSMM <: AbstractLatentStateModel
        @test !(AbstractHMM <: AbstractHSMM)
        @test !(AbstractHSMM <: AbstractHMM)
        @test hmm isa AbstractLatentStateModel
        @test !(hmm isa AbstractHSMM)
    end

    @testset "HMM does not gain a duration_distributions method" begin
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

@testset "HSMM joint_logdensityof" begin
    init = [0.6, 0.4]
    trans = [0.0 1.0; 1.0 0.0]
    dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
    dur_dists = [Geometric(0.4), Geometric(0.6)]

    @testset "Hand-computed value" begin
        hsmm = HSMM(init, trans, dists, dur_dists)
        # states [1, 1, 2] = one full segment (state 1, length 2), then a censored
        # final segment (state 2, length 1 observed).
        state_seq = [1, 1, 2]
        obs_seq = [0.3, -0.1, 5.2]
        expected =
            log(init[1]) +
            duration_logdensityof(dur_dists[1], 2) +
            log(trans[1, 2]) +
            duration_logsurvival(dur_dists[2], 1) +
            logdensityof(dists[1], obs_seq[1]) +
            logdensityof(dists[1], obs_seq[2]) +
            logdensityof(dists[2], obs_seq[3])
        @test joint_logdensityof(hsmm, obs_seq, state_seq) ≈ expected
    end

    @testset "Survival term" begin
        d = Geometric(0.4)
        # P(D >= 1) == 1 for any sojourn distribution.
        @test duration_logsurvival(d, 1) ≈ 0.0
        # P(D >= k) == 1 - sum_{j < k} P(D == j), by construction.
        for k in 1:6
            head = sum(exp(duration_logdensityof(d, j)) for j in 1:(k - 1); init=0.0)
            @test duration_logsurvival(d, k) ≈ log1p(-head)
        end
        # Geometric sojourns have the closed form P(D >= k) == (1 - p)^(k - 1).
        for k in 1:6
            @test duration_logsurvival(d, k) ≈ (k - 1) * log(1 - 0.4)
        end
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        #= An HMM with self-transition probability A[i, i] is exactly an HSMM with
        geometric sojourns and the self-transitions renormalized away. The equality
        below holds only because the final segment is right-censored. =#
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm_init = [0.5, 0.3, 0.2]
        hmm_dists = [Normal(0.0), Normal(5.0), Normal(10.0)]
        hmm = HMM(hmm_init, hmm_trans, hmm_dists)

        N = length(hmm_init)
        hsmm_trans = [
            i == j ? 0.0 : hmm_trans[i, j] / (1 - hmm_trans[i, i]) for i in 1:N, j in 1:N
        ]
        hsmm_durs = [Geometric(1 - hmm_trans[i, i]) for i in 1:N]
        hsmm = HSMM(hmm_init, hsmm_trans, hmm_dists, hsmm_durs)

        rng = StableRNG(63)
        for _ in 1:20
            (; state_seq, obs_seq) = rand(rng, hmm, 12)
            @test joint_logdensityof(hsmm, obs_seq, state_seq) ≈
                joint_logdensityof(hmm, obs_seq, state_seq)
        end
    end

    @testset "Multiple sequences" begin
        hsmm = HSMM(init, trans, dists, dur_dists)
        rng = StableRNG(15)
        sims = [rand(rng, hsmm, T) for T in (4, 7, 5)]
        obs_seq = reduce(vcat, sim.obs_seq for sim in sims)
        state_seq = reduce(vcat, sim.state_seq for sim in sims)
        seq_ends = cumsum([length(sim.obs_seq) for sim in sims])
        @test joint_logdensityof(hsmm, obs_seq, state_seq; seq_ends) ≈
            sum(joint_logdensityof(hsmm, sim.obs_seq, sim.state_seq) for sim in sims)
    end

    @testset "Type promotion" begin
        hsmm32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            dur_dists,
        )
        # Float32 model with Float64 duration distributions promotes to Float64.
        @test joint_logdensityof(hsmm32, [0.0f0, 5.0f0], [1, 2]) isa Float64
    end
end
