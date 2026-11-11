using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using HiddenMarkovModels
using HiddenMarkovModels:
    AbstractLatentStateModel,
    AbstractHSMM,
    HSMMForwardBackwardStorage,
    HSMMForwardStorage,
    HSMMViterbiStorage,
    duration_distributions,
    duration_logdensityof,
    duration_logsurvival,
    duration_logdensity_type,
    elementwise_log,
    Fill,
    StateSegments,
    fit_duration_in_sequence!,
    forward!,
    forward_backward!,
    initialize_forward,
    initialize_forward_backward,
    initialize_viterbi,
    log_initialization,
    log_transition_matrix,
    valid_hsmm,
    viterbi!
using Distributions: Binomial, Categorical, Geometric, Normal, Poisson, mean, std, succprob
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using JET: @test_call, @test_opt
using Random: AbstractRNG
using SparseArrays: SparseMatrixCSC, nnz, sparse
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

    @testset "Integer transition matrix" begin
        @test HSMM(init, [0 1; 1 0], dists, dur_dists) isa HSMM
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

# Summing over the iterator inside a function so `@allocated` measures only the iteration.
function total_span(state_seq, t1, t2)
    return sum(t_end - t_start + 1 for (t_start, t_end) in StateSegments(state_seq, t1, t2))
end

@testset "State segments" begin
    @test collect(StateSegments([1, 1, 2, 2, 2, 1], 1, 6)) == [(1, 2), (3, 5), (6, 6)]
    @test collect(StateSegments([1, 1, 2, 2, 2, 1], 2, 4)) == [(2, 2), (3, 4)]
    @test collect(StateSegments([3], 1, 1)) == [(1, 1)]
    @test isempty(collect(StateSegments([1, 2], 2, 1)))
    @test eltype(StateSegments([1, 2], 1, 2)) == Tuple{Int,Int}
    state_seq = [1, 1, 2, 2, 2, 1]
    @test total_span(state_seq, 1, 6) == 6
    @test iszero(@allocated total_span(state_seq, 1, 6))
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

# A duck-typed equivalent of `Geometric(p)`.
struct DuckGeometric
    p::Float64
end
DensityInterface.DensityKind(::DuckGeometric) = HasDensity()
function DensityInterface.logdensityof(d::DuckGeometric, k)
    return k < 0 ? -Inf : log(d.p) + k * log1p(-d.p)
end
Base.rand(::AbstractRNG, ::DuckGeometric) = 0
function HiddenMarkovModels.duration_logsurvival(d::DuckGeometric, k::Integer)
    return k <= 1 ? 0.0 : (k - 1) * log1p(-d.p)
end

# The control at the start of a segment selects its duration distribution.
struct ControlledDurationHSMM <: AbstractHSMM
    init::Vector{Float64}
    trans::Matrix{Float64}
    dists::Vector{Normal{Float64}}
    durs::Vector{Vector{Geometric{Float64}}}
end

HiddenMarkovModels.initialization(hsmm::ControlledDurationHSMM) = hsmm.init
HiddenMarkovModels.transition_matrix(hsmm::ControlledDurationHSMM) = hsmm.trans
HiddenMarkovModels.transition_matrix(hsmm::ControlledDurationHSMM, ::Integer) = hsmm.trans
HiddenMarkovModels.obs_distributions(hsmm::ControlledDurationHSMM) = hsmm.dists
HiddenMarkovModels.obs_distributions(hsmm::ControlledDurationHSMM, ::Integer) = hsmm.dists
HiddenMarkovModels.duration_distributions(hsmm::ControlledDurationHSMM) = hsmm.durs[1]
function HiddenMarkovModels.duration_distributions(
    hsmm::ControlledDurationHSMM, control::Integer
)
    return hsmm.durs[control]
end

# Every component depends on the control, so using the wrong control changes the results.
struct FullyControlledHSMM <: AbstractHSMM
    init::Vector{Float64}
    trans::Vector{Matrix{Float64}}
    dists::Vector{Vector{Normal{Float64}}}
    durs::Vector{Vector{Geometric{Float64}}}
end

HiddenMarkovModels.initialization(hsmm::FullyControlledHSMM) = hsmm.init
HiddenMarkovModels.transition_matrix(hsmm::FullyControlledHSMM) = hsmm.trans[1]
function HiddenMarkovModels.transition_matrix(hsmm::FullyControlledHSMM, control::Integer)
    return hsmm.trans[control]
end
HiddenMarkovModels.obs_distributions(hsmm::FullyControlledHSMM) = hsmm.dists[1]
function HiddenMarkovModels.obs_distributions(hsmm::FullyControlledHSMM, control::Integer)
    return hsmm.dists[control]
end
HiddenMarkovModels.duration_distributions(hsmm::FullyControlledHSMM) = hsmm.durs[1]
function HiddenMarkovModels.duration_distributions(
    hsmm::FullyControlledHSMM, control::Integer
)
    return hsmm.durs[control]
end

function fully_controlled_hsmm()
    init = [0.5, 0.3, 0.2]
    trans = [
        [0.0 0.9 0.1; 0.2 0.0 0.8; 0.6 0.4 0.0], [0.0 0.1 0.9; 0.8 0.0 0.2; 0.3 0.7 0.0]
    ]
    dists = [
        [Normal(0.0), Normal(2.0), Normal(4.0)], [Normal(4.0), Normal(0.0), Normal(2.0)]
    ]
    durs = [
        [Geometric(0.2), Geometric(0.5), Geometric(0.8)],
        [Geometric(0.7), Geometric(0.3), Geometric(0.4)],
    ]
    return FullyControlledHSMM(init, trans, dists, durs)
end

#= Sojourns last at most 2 timesteps and state 2 almost never emits `1`, so a self-transition of
probability `1e-17` in state 1 would explain `[1, 1, 1, 1]` better than the valid paths. =#
const SELF_TRANSITION_OBS = [1, 1, 1, 1]
function self_transition_hsmm(trans)
    dists = [Categorical([1.0, 0.0]), Categorical([1e-20, 1 - 1e-20])]
    return HSMM([0.6, 0.4], trans, dists, [Binomial(1, 0.5), Binomial(1, 0.5)])
end
const NEAR_ZERO_DIAGONALS = ([1e-17 1.0; 1.0 1e-17], sparse([1e-17 1.0; 1.0 1e-17]))

all_state_seqs(N, T) = (collect(s) for s in Iterators.product(ntuple(_ -> 1:N, T)...))

function brute_force_logdensityof(hsmm, obs_seq, control_seq, N)
    T = length(obs_seq)
    return Float64(
        log(
            sum(
                exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq))) for
                s in all_state_seqs(N, T)
            ),
        ),
    )
end

function brute_force_marginals(hsmm, obs_seq, control_seq, N)
    T = length(obs_seq)
    α = zeros(BigFloat, N, T)
    for t in 1:T
        for s in all_state_seqs(N, t)
            α[s[t], t] += exp(
                big(joint_logdensityof(hsmm, obs_seq[1:t], s, control_seq[1:t]))
            )
        end
        α[:, t] ./= sum(α[:, t])
    end
    return Float64.(α)
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

function rand_hsmm(rng, N)
    init = rand(rng, N)
    init ./= sum(init)
    trans = [i == j ? 0.0 : rand(rng) for i in 1:N, j in 1:N]
    for i in 1:N
        trans[i, :] ./= sum(trans[i, :])
    end
    dists = [Normal(2.0 * i, 1.0) for i in 1:N]
    dur_dists = [Geometric(0.2 + 0.15 * i) for i in 1:N]
    return HSMM(init, trans, dists, dur_dists)
end

@testset "HSMM forward" begin
    @testset "Brute-force marginalization over all state sequences" begin
        rng = StableRNG(101)
        for N in 2:3, T in 5:7
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            @test logdensityof(hsmm, obs_seq; max_duration=T) ≈
                brute_force_logdensityof(hsmm, obs_seq, control_seq, N)
        end
    end

    @testset "Normalization over all observation sequences" begin
        # This relies on right-censoring the final segment.
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8])]
        dur_dists = [Geometric(0.4), Geometric(0.55)]
        hsmm = HSMM(init, trans, dists, dur_dists)
        T = 4
        total = sum(
            exp(logdensityof(hsmm, collect(obs))) for
            obs in Iterators.product(ntuple(_ -> 1:2, T)...)
        )
        @test total ≈ 1.0

        init3 = [0.5, 0.3, 0.2]
        trans3 = [0.0 0.5 0.5; 0.4 0.0 0.6; 0.7 0.3 0.0]
        dists3 = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8]), Categorical([0.5, 0.5])]
        dur3 = [Geometric(0.3), Geometric(0.5), Geometric(0.9)]
        hsmm3 = HSMM(init3, trans3, dists3, dur3)
        total3 = sum(
            exp(logdensityof(hsmm3, collect(obs))) for
            obs in Iterators.product(ntuple(_ -> 1:2, T)...)
        )
        @test total3 ≈ 1.0
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        hmm_init = [0.5, 0.3, 0.2]
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm_dists = [Normal(0.0), Normal(5.0), Normal(10.0)]
        hmm = HMM(hmm_init, hmm_trans, hmm_dists)

        N = length(hmm_init)
        hsmm_trans = [
            i == j ? 0.0 : hmm_trans[i, j] / (1 - hmm_trans[i, i]) for i in 1:N, j in 1:N
        ]
        hsmm_durs = [Geometric(1 - hmm_trans[i, i]) for i in 1:N]
        hsmm = HSMM(hmm_init, hsmm_trans, hmm_dists, hsmm_durs)

        rng = StableRNG(64)
        T = 12
        for _ in 1:10
            (; obs_seq) = rand(rng, hmm, T)
            @test logdensityof(hsmm, obs_seq; max_duration=T) ≈ logdensityof(hmm, obs_seq)
        end

        # Regression test for precision loss when survival is computed from the head complement.
        long_trans = [0.8 0.12 0.08; 0.1 0.85 0.05; 0.15 0.1 0.75]
        long_hmm = HMM(
            [0.4, 0.35, 0.25], long_trans, [Normal(2.0), Normal(4.0), Normal(6.0)]
        )
        long_hsmm = HSMM(
            [0.4, 0.35, 0.25],
            [
                i == j ? 0.0 : long_trans[i, j] / (1 - long_trans[i, i]) for i in 1:N,
                j in 1:N
            ],
            [Normal(2.0), Normal(4.0), Normal(6.0)],
            [Geometric(1 - long_trans[i, i]) for i in 1:N],
        )
        long_obs = randn(StableRNG(65), 500)
        @test logdensityof(long_hsmm, long_obs) ≈ logdensityof(long_hmm, long_obs)
    end

    @testset "Multiple sequences of differing lengths" begin
        # Regression test for a race in the observation prefix sums.
        rng = StableRNG(15)
        hsmm = rand_hsmm(rng, 3)
        lengths = (4, 11, 7, 9, 5, 13)
        obs_seqs = [randn(rng, T) for T in lengths]
        obs_seq = reduce(vcat, obs_seqs)
        seq_ends = cumsum(collect(lengths))
        joint = logdensityof(hsmm, obs_seq; seq_ends)
        separate = sum(
            logdensityof(hsmm, o; max_duration=maximum(lengths)) for o in obs_seqs
        )
        @test joint ≈ separate
    end

    @testset "Controlled duration distributions" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(4.0, 1.0)]
        durs = [[Geometric(0.15), Geometric(0.25)], [Geometric(0.85), Geometric(0.75)]]
        hsmm = ControlledDurationHSMM(init, trans, dists, durs)
        @test duration_distributions(hsmm, 1) != duration_distributions(hsmm, 2)

        rng = StableRNG(202)
        T = 6
        obs_seq = randn(rng, T)
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1])
            @test logdensityof(hsmm, obs_seq, control_seq; max_duration=T) ≈
                brute_force_logdensityof(hsmm, obs_seq, control_seq, 2)
            α, _ = forward(hsmm, obs_seq, control_seq; max_duration=T)
            @test α ≈ brute_force_marginals(hsmm, obs_seq, control_seq, 2)
        end
        @test logdensityof(hsmm, obs_seq, [1, 1, 2, 2, 1, 1]) !=
            logdensityof(hsmm, obs_seq, [2, 2, 1, 1, 2, 2])

        # A `Fill` of a real control takes the fast path for constant controls.
        for c in 1:2
            expected = brute_force_logdensityof(hsmm, obs_seq, fill(c, T), 2)
            @test logdensityof(hsmm, obs_seq, Fill(c, T)) ≈ expected
            @test logdensityof(hsmm, obs_seq, fill(c, T)) ≈ expected
            @test forward(hsmm, obs_seq, Fill(c, T))[1] ≈
                brute_force_marginals(hsmm, obs_seq, fill(c, T), 2)
        end
    end

    @testset "Controlled transitions, durations and observations" begin
        hsmm = fully_controlled_hsmm()
        rng = StableRNG(203)
        T = 6
        obs_seq = 2 .* randn(rng, T) .+ 2
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], rand(rng, 1:2, T))
            @test logdensityof(hsmm, obs_seq, control_seq) ≈
                brute_force_logdensityof(hsmm, obs_seq, control_seq, 3)
            @test forward(hsmm, obs_seq, control_seq)[1] ≈
                brute_force_marginals(hsmm, obs_seq, control_seq, 3)
        end
    end

    @testset "Sparse transition matrix" begin
        # Structural zeros must be read as impossible transitions, not as `log(1)`.
        init = [0.5, 0.3, 0.2]
        trans = [0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0]
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = [Geometric(0.3) for _ in 1:3]
        hsmm_dense = HSMM(init, trans, dists, durs)
        hsmm_sparse = HSMM(init, sparse(trans), dists, durs)
        rng = StableRNG(1)
        for T in (6, 10)
            obs_seq = 3 .* randn(rng, T)
            α_dense, logL_dense = forward(hsmm_dense, obs_seq)
            α_sparse, logL_sparse = forward(hsmm_sparse, obs_seq)
            @test logL_sparse ≈ logL_dense
            @test α_sparse ≈ α_dense
        end
        obs_seq = 3 .* randn(rng, 6)
        @test logdensityof(hsmm_sparse, obs_seq) ≈
            brute_force_logdensityof(hsmm_sparse, obs_seq, fill(nothing, 6), 3)

        # The path `2 → 3` uses a forbidden transition.
        obs_seq = [3.0, 6.0, 0.0]
        for state_seq in ([2, 3, 1], [1, 2, 1])
            @test joint_logdensityof(hsmm_sparse, obs_seq, state_seq) ==
                joint_logdensityof(hsmm_dense, obs_seq, state_seq)
        end
        @test joint_logdensityof(hsmm_sparse, obs_seq, [2, 3, 1]) == -Inf
    end

    @testset "Bounded duration support" begin
        # Sojourns last at most 3 timesteps, so longer segments are skipped.
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0), Normal(3.0)]
        hsmm = HSMM(init, trans, dists, [Binomial(2, 0.4), Binomial(1, 0.7)])
        rng = StableRNG(12)
        T = 7
        obs_seq = 3 .* randn(rng, T)
        control_seq = fill(nothing, T)
        exact = logdensityof(hsmm, obs_seq)
        @test exact ≈ brute_force_logdensityof(hsmm, obs_seq, control_seq, 2)
        @test forward(hsmm, obs_seq)[1] ≈
            brute_force_marginals(hsmm, obs_seq, control_seq, 2)
        for max_duration in 3:T
            @test logdensityof(hsmm, obs_seq; max_duration) == exact
        end
        @test logdensityof(hsmm, obs_seq; max_duration=2) < exact
    end

    @testset "Duck-typed duration distributions" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
        hsmm_ref = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        hsmm_duck = HSMM(init, trans, dists, [DuckGeometric(0.4), DuckGeometric(0.6)])
        obs_seq = randn(StableRNG(9), 8)
        @test logdensityof(hsmm_duck, obs_seq) ≈ logdensityof(hsmm_ref, obs_seq)
    end

    @testset "Near-zero diagonal is ignored" begin
        # `valid_hsmm` tolerates diagonal entries up to `eps`.
        init = [0.6, 0.4]
        dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
        durs = [Geometric(0.4), Geometric(0.6)]
        hsmm_zero = HSMM(init, [0.0 1.0; 1.0 0.0], dists, durs)
        hsmm_eps = HSMM(init, [1e-17 1.0; 1.0 1e-17], dists, durs)
        obs_seq = randn(StableRNG(10), 8)
        @test forward(hsmm_eps, obs_seq) == forward(hsmm_zero, obs_seq)

        # `sparse` drops exact zeros but stores the near-zero diagonal.
        α_zero, logL_zero = forward(
            self_transition_hsmm([0.0 1.0; 1.0 0.0]), SELF_TRANSITION_OBS
        )
        for trans in NEAR_ZERO_DIAGONALS
            α, logL = forward(self_transition_hsmm(trans), SELF_TRANSITION_OBS)
            @test α ≈ α_zero
            @test logL ≈ logL_zero
        end
    end

    @testset "Type promotion" begin
        hsmm32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4), Geometric(0.6)],
        )
        obs_seq = Float32[0.0, 5.0, 0.1, 4.9]
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_forward(
            hsmm32, obs_seq, control_seq; seq_ends=(length(obs_seq),)
        )
        @test storage isa HSMMForwardStorage{Float64}
        @test logdensityof(hsmm32, obs_seq) isa Float64

        hsmm_all32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4f0), Geometric(0.6f0)],
        )
        storage32 = initialize_forward(
            hsmm_all32, obs_seq, control_seq; seq_ends=(length(obs_seq),)
        )
        @test storage32 isa HSMMForwardStorage{Float32}
    end

    @testset "error_if_not_finite" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        obs_seq = [1, 2, 3]  # 3 is outside the support of every observation distribution
        @test logdensityof(hsmm, obs_seq) == -Inf
        @test_throws ArgumentError forward(hsmm, obs_seq)
        @test_throws ArgumentError forward(hsmm, obs_seq; error_if_not_finite=true)
        _, logL = forward(hsmm, obs_seq; error_if_not_finite=false)
        @test only(logL) == -Inf
    end

    @testset "Zero-probability observation before the last timestep" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.0, 1.0]), Categorical([0.5, 0.5])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        for obs_seq in ([1, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2, 2, 2])
            control_seq = fill(nothing, length(obs_seq))
            expected = brute_force_logdensityof(hsmm, obs_seq, control_seq, 2)
            @test isfinite(expected)
            @test logdensityof(hsmm, obs_seq) ≈ expected
            @test forward(hsmm, obs_seq)[1] ≈
                brute_force_marginals(hsmm, obs_seq, control_seq, 2)
        end
        @test logdensityof(hsmm, [1, 2, 2]) ≈ -1.83258146374831

        # The impossible emission only affects segments that contain it.
        α, _ = forward(hsmm, [1, 2, 2])
        @test α[:, 1] ≈ [0.0, 1.0]
        @test α[:, 2] ≈ [0.75, 0.25]
    end

    @testset "max_duration" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0), Normal(4.0)]
        # Long sojourns make truncation visible.
        hsmm = HSMM(init, trans, dists, [Geometric(0.08), Geometric(0.12)])
        T = 12
        obs_seq = rand(StableRNG(2), hsmm, T).obs_seq

        exact = logdensityof(hsmm, obs_seq)
        @test exact == logdensityof(hsmm, obs_seq; max_duration=T)
        for extra in 1:5
            @test logdensityof(hsmm, obs_seq; max_duration=T + extra) == exact
        end
        errors = [abs(logdensityof(hsmm, obs_seq; max_duration=d) - exact) for d in 1:T]
        @test issorted(errors; rev=true)
        for d in 1:(T - 1)
            @test logdensityof(hsmm, obs_seq; max_duration=d) < exact
        end
        @test errors[1] > 1e-3
        @test errors[T] == 0
        @test_throws ArgumentError logdensityof(hsmm, obs_seq; max_duration=0)
    end

    @testset "Filtered state marginals" begin
        rng = StableRNG(101)
        hsmm = rand_hsmm(rng, 3)
        T = 25
        obs_seq = rand(rng, hsmm, T).obs_seq
        α, logL = forward(hsmm, obs_seq)

        @test size(α) == (3, T)
        @test all(>=(0), α)
        for t in 1:T
            @test sum(α[:, t]) ≈ 1
        end
        @test logL[1] ≈ logdensityof(hsmm, obs_seq)

        # Check every prefix against brute-force marginalization.
        rng = StableRNG(303)
        for N in 2:3, T in 5:7
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            storage = initialize_forward(
                hsmm, obs_seq, control_seq; seq_ends=(T,), max_duration=T
            )
            forward!(storage, hsmm, obs_seq, control_seq; seq_ends=(T,))
            expected = brute_force_marginals(hsmm, obs_seq, control_seq, N)
            # `log_prefix` is an internal buffer, checked here to cover every prefix at once.
            for t in 1:T
                @test storage.log_prefix[t] ≈
                    brute_force_logdensityof(hsmm, obs_seq[1:t], control_seq[1:t], N)
                @test storage.α[:, t] ≈ expected[:, t]
            end
        end
    end

    @testset "Filtered marginals match the equivalent HMM exactly" begin
        hmm_init = [0.5, 0.3, 0.2]
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm_dists = [Normal(0.0), Normal(5.0), Normal(10.0)]
        hmm = HMM(hmm_init, hmm_trans, hmm_dists)

        N = length(hmm_init)
        hsmm_trans = [
            i == j ? 0.0 : hmm_trans[i, j] / (1 - hmm_trans[i, i]) for i in 1:N, j in 1:N
        ]
        hsmm_durs = [Geometric(1 - hmm_trans[i, i]) for i in 1:N]
        hsmm = HSMM(hmm_init, hsmm_trans, hmm_dists, hsmm_durs)

        rng = StableRNG(102)
        for _ in 1:10
            obs_seq = rand(rng, hmm, 15).obs_seq
            α_hmm, logL_hmm = forward(hmm, obs_seq)
            α_hsmm, logL_hsmm = forward(hsmm, obs_seq; max_duration=15)
            @test α_hsmm ≈ α_hmm
            @test logL_hsmm ≈ logL_hmm
        end
    end

    @testset "Empty subsequences" begin
        hsmm = rand_hsmm(StableRNG(5), 2)
        obs_seq = randn(StableRNG(6), 5)
        expected = logdensityof(hsmm, obs_seq)
        @test logdensityof(hsmm, obs_seq; seq_ends=[0, 5]) ≈ expected
        @test logdensityof(hsmm, obs_seq; seq_ends=[5, 5]) ≈ expected
        @test logdensityof(hsmm, obs_seq; seq_ends=[2, 2, 5]) ≈
            logdensityof(hsmm, obs_seq; seq_ends=[2, 5])
    end

    @testset "ForwardDiff gradients with impossible terms" begin
        # Each model makes `-Inf` terms reach the log-domain recursion.
        fd_gradient(f, θ) = only(FiniteDifferences.grad(central_fdm(5, 1), f, θ))
        obs_seq = randn(StableRNG(7), 20) .* 2 .+ 3
        function zero_off_diagonal(θ)
            trans = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
            dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
            return HSMM(fill(1 / 3, 3), trans, dists, [Geometric(θ[3]) for _ in 1:3])
        end
        function zero_init(θ)
            init = [one(θ[1]), zero(θ[1]), zero(θ[1])]
            trans = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0] .* θ[3] ./ θ[3]
            dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
            return HSMM(init, trans, dists, [Geometric(0.3) for _ in 1:3])
        end
        function bounded_durations(θ)
            trans = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0]
            dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
            return HSMM(fill(1 / 3, 3), trans, dists, [Binomial(4, θ[3]) for _ in 1:3])
        end
        θ = [0.0, 3.0, 0.4]
        for build in (zero_off_diagonal, zero_init, bounded_durations)
            f(θ) = logdensityof(build(θ), obs_seq)
            g = ForwardDiff.gradient(f, θ)
            @test all(isfinite, g)
            @test g ≈ fd_gradient(f, θ) rtol = 1e-5
        end

        # Here `b[1,t] = 0` whenever `obs_seq[t] = 1`, but as a `Dual`, so its log has value
        # `-Inf` with NaN partials. It must still be caught by the zero count of the prefix sums.
        function impossible_obs(θ)
            trans = [0.0 1.0; 1.0 0.0]
            dists = [
                Categorical([zero(θ[1]), θ[1], 1 - θ[1]]), Categorical([0.3, 0.3, 0.4])
            ]
            return HSMM([0.5, 0.5], trans, dists, [Geometric(θ[2]), Geometric(0.5)])
        end
        f_obs(θ) = logdensityof(impossible_obs(θ), [1, 2, 3, 3, 1, 2])
        g = ForwardDiff.gradient(f_obs, [0.4, 0.3])
        @test all(isfinite, g)
        @test g ≈ fd_gradient(f_obs, [0.4, 0.3]) rtol = 1e-5
    end

    @testset "Allocations" begin
        rng = StableRNG(77)
        hsmm = rand_hsmm(rng, 3)
        T = 30
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        # An `NTuple{1}` for `seq_ends` disables multithreading, as in the HMM allocation test.
        seq_ends = (T,)
        storage = initialize_forward(hsmm, obs_seq, control_seq; seq_ends)
        call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)  # warm up
        @test (@allocated call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)) == 0
    end
end

function longest_segment(state_seq)
    return maximum(
        t_end - t_start + 1 for
        (t_start, t_end) in StateSegments(state_seq, 1, length(state_seq))
    )
end

function brute_force_posteriors(hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq))
    T = length(obs_seq)
    γ = zeros(BigFloat, N, T)
    ξ = [zeros(BigFloat, N, N) for _ in 1:(T - 1)]
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        p = exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq)))
        for t in 1:T
            γ[s[t], t] += p
        end
        for t in 1:(T - 1)
            if s[t] != s[t + 1]
                ξ[t][s[t], s[t + 1]] += p
            end
        end
    end
    Z = sum(γ[:, 1])
    return Float64.(γ ./ Z), [Float64.(ξₜ ./ Z) for ξₜ in ξ], Float64(log(Z))
end

function full_forward_backward(
    hsmm, obs_seq, control_seq=fill(nothing, length(obs_seq)); kwargs...
)
    seq_ends = get(kwargs, :seq_ends, (length(obs_seq),))
    storage = initialize_forward_backward(hsmm, obs_seq, control_seq; kwargs...)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return storage
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward_backward!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

@testset "HSMM forward-backward" begin
    @testset "Brute-force marginalization over all state sequences" begin
        rng = StableRNG(401)
        for N in 2:3, T in 5:7
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            γ_bf, ξ_bf, logL_bf = brute_force_posteriors(hsmm, obs_seq, control_seq, N)
            storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,))
            @test storage.γ ≈ γ_bf
            @test storage.ξ[1:(T - 1)] ≈ ξ_bf
            @test iszero(storage.ξ[T])
            @test only(storage.logL) ≈ logL_bf

            γ, logL = forward_backward(hsmm, obs_seq)
            @test γ ≈ γ_bf
            @test logL ≈ storage.logL
        end
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        hmm_init = [0.5, 0.3, 0.2]
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm_dists = [Normal(0.0), Normal(5.0), Normal(10.0)]
        hmm = HMM(hmm_init, hmm_trans, hmm_dists)

        N = length(hmm_init)
        hsmm_trans = [
            i == j ? 0.0 : hmm_trans[i, j] / (1 - hmm_trans[i, i]) for i in 1:N, j in 1:N
        ]
        hsmm_durs = [Geometric(1 - hmm_trans[i, i]) for i in 1:N]
        hsmm = HSMM(hmm_init, hsmm_trans, hmm_dists, hsmm_durs)

        # A long sequence also checks that the log-domain recursion does not lose precision.
        for (seed, T) in ((1, 15), (2, 15), (3, 500))
            obs_seq = rand(StableRNG(seed), hmm, T).obs_seq
            control_seq = fill(nothing, T)
            storage_hmm = initialize_forward_backward(
                hmm, obs_seq, control_seq; seq_ends=(T,)
            )
            forward_backward!(storage_hmm, hmm, obs_seq, control_seq; seq_ends=(T,))
            storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,))
            @test storage.γ ≈ storage_hmm.γ
            @test storage.logL ≈ storage_hmm.logL
            # The HMM self-transitions are the continued sojourns of the HSMM.
            off_diagonal = [i != j for i in 1:N, j in 1:N]
            for t in 1:(T - 1)
                @test storage.ξ[t] ≈ storage_hmm.ξ[t] .* off_diagonal
            end

            γ_hmm, logL_hmm = forward_backward(hmm, obs_seq)
            γ, logL = forward_backward(hsmm, obs_seq)
            @test γ ≈ γ_hmm
            @test logL ≈ logL_hmm
        end
    end

    @testset "Out-of-place versions agree with the in-place ones" begin
        rng = StableRNG(406)
        hsmm = rand_hsmm(rng, 3)
        lengths = (6, 1, 9)
        obs_seq = randn(rng, sum(lengths))
        seq_ends = cumsum(collect(lengths))
        control_seq = fill(nothing, length(obs_seq))
        storage = full_forward_backward(hsmm, obs_seq; seq_ends)
        forward_storage = initialize_forward(hsmm, obs_seq, control_seq; seq_ends)
        forward!(forward_storage, hsmm, obs_seq, control_seq; seq_ends)

        γ, logL = forward_backward(hsmm, obs_seq; seq_ends)
        @test γ ≈ storage.γ
        @test logL ≈ storage.logL
        α, logL_forward = forward(hsmm, obs_seq; seq_ends)
        @test α ≈ forward_storage.α
        @test logL_forward ≈ forward_storage.logL
        @test logL_forward ≈ logL
        # Smoothing agrees with filtering at the end of each sequence.
        @test γ[:, seq_ends] ≈ α[:, seq_ends]
    end

    @testset "Marginal consistency" begin
        rng = StableRNG(402)
        hsmm = rand_hsmm(rng, 3)
        T = 40
        obs_seq = rand(rng, hsmm, T).obs_seq
        storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,))
        (; γ, ξ) = storage
        @test all(>=(0), γ)
        for t in 1:T
            @test sum(γ[:, t]) ≈ 1
        end
        # The mass of a state changes only through segment boundaries.
        for t in 1:(T - 1)
            @test γ[:, t + 1] - γ[:, t] ≈ vec(sum(ξ[t]; dims=1)) - vec(sum(ξ[t]; dims=2)) atol =
                1e-12
        end
        # Smoothing agrees with filtering at the last timestep.
        @test γ[:, T] ≈ storage.forward.α[:, T]
    end

    @testset "Controlled duration distributions" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(4.0, 1.0)]
        durs = [[Geometric(0.15), Geometric(0.25)], [Geometric(0.85), Geometric(0.75)]]
        hsmm = ControlledDurationHSMM(init, trans, dists, durs)

        T = 6
        obs_seq = randn(StableRNG(403), T)
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1])
            γ_bf, ξ_bf, logL_bf = brute_force_posteriors(hsmm, obs_seq, control_seq, 2)
            storage = full_forward_backward(hsmm, obs_seq, control_seq; seq_ends=(T,))
            @test storage.γ ≈ γ_bf
            @test storage.ξ[1:(T - 1)] ≈ ξ_bf
            @test only(storage.logL) ≈ logL_bf

            γ, logL = forward_backward(hsmm, obs_seq, control_seq)
            @test γ ≈ γ_bf
            @test only(logL) ≈ logL_bf
        end
    end

    @testset "Controlled transitions, durations and observations" begin
        hsmm = fully_controlled_hsmm()
        rng = StableRNG(410)
        T = 6
        obs_seq = 2 .* randn(rng, T) .+ 2
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], rand(rng, 1:2, T))
            γ_bf, ξ_bf, logL_bf = brute_force_posteriors(hsmm, obs_seq, control_seq, 3)
            storage = full_forward_backward(hsmm, obs_seq, control_seq; seq_ends=(T,))
            @test storage.γ ≈ γ_bf
            @test storage.ξ[1:(T - 1)] ≈ ξ_bf
            @test only(storage.logL) ≈ logL_bf
        end
    end

    @testset "Near-zero diagonal is ignored" begin
        # `valid_hsmm` tolerates diagonal entries up to `eps`.
        obs_seq = SELF_TRANSITION_OBS
        seq_ends = (length(obs_seq),)
        hsmm_zero = self_transition_hsmm([0.0 1.0; 1.0 0.0])
        storage_zero = full_forward_backward(hsmm_zero, obs_seq; seq_ends)
        # `sparse` drops exact zeros but stores the near-zero diagonal.
        for trans in NEAR_ZERO_DIAGONALS
            storage = full_forward_backward(self_transition_hsmm(trans), obs_seq; seq_ends)
            @test storage.γ ≈ storage_zero.γ
            @test storage.ξ ≈ storage_zero.ξ
            @test storage.logL ≈ storage_zero.logL
        end
    end

    @testset "Sparse transition matrix" begin
        # Structural zeros must be read as impossible transitions, not as `log(1)`.
        init = [0.5, 0.3, 0.2]
        trans = [0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0]
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = [Geometric(0.3) for _ in 1:3]
        hsmm_dense = HSMM(init, trans, dists, durs)
        hsmm_sparse = HSMM(init, sparse(trans), dists, durs)
        rng = StableRNG(407)
        T = 6
        obs_seq = 3 .* randn(rng, T)
        control_seq = fill(nothing, T)
        γ_bf, ξ_bf, logL_bf = brute_force_posteriors(hsmm_sparse, obs_seq, control_seq, 3)
        storage_dense = full_forward_backward(hsmm_dense, obs_seq; seq_ends=(T,))
        storage = full_forward_backward(hsmm_sparse, obs_seq; seq_ends=(T,))
        @test storage.γ ≈ γ_bf
        @test storage.γ ≈ storage_dense.γ
        @test storage.ξ[1:(T - 1)] ≈ ξ_bf
        @test only(storage.logL) ≈ logL_bf
        for t in 1:(T - 1)
            # The forbidden transitions keep their structural zeros.
            @test storage.ξ[t] isa SparseMatrixCSC
            @test nnz(storage.ξ[t]) == nnz(sparse(trans))
            @test storage.ξ[t] ≈ storage_dense.ξ[t]
        end
        @test forward_backward(hsmm_sparse, obs_seq)[1] ≈ γ_bf

        seq_ends = (T,)
        call_forward_backward!(storage, hsmm_sparse, obs_seq, control_seq, seq_ends)
        @test (@allocated call_forward_backward!(
            storage, hsmm_sparse, obs_seq, control_seq, seq_ends
        )) == 0
    end

    @testset "Bounded duration support" begin
        # Sojourns last at most 3 timesteps, so longer segments are skipped.
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0), Normal(3.0)]
        hsmm = HSMM(init, trans, dists, [Binomial(2, 0.4), Binomial(1, 0.7)])
        rng = StableRNG(408)
        T = 7
        obs_seq = 3 .* randn(rng, T)
        control_seq = fill(nothing, T)
        γ_bf, ξ_bf, logL_bf = brute_force_posteriors(hsmm, obs_seq, control_seq, 2)
        storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,))
        @test storage.γ ≈ γ_bf
        @test storage.ξ[1:(T - 1)] ≈ ξ_bf
        @test only(storage.logL) ≈ logL_bf
        for max_duration in 3:T
            @test forward_backward(hsmm, obs_seq; max_duration)[1] ≈ γ_bf
        end
    end

    @testset "max_duration" begin
        rng = StableRNG(404)
        T = 7
        for N in 2:3, max_duration in 1:4
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            γ_bf, ξ_bf, logL_bf = brute_force_posteriors(
                hsmm, obs_seq, control_seq, N; max_duration
            )
            storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,), max_duration)
            @test storage.γ ≈ γ_bf
            @test storage.ξ[1:(T - 1)] ≈ ξ_bf
            @test only(storage.logL) ≈ logL_bf
            @test only(storage.logL) ≈ logdensityof(hsmm, obs_seq; max_duration)

            γ, logL = forward_backward(hsmm, obs_seq; max_duration)
            @test γ ≈ γ_bf
            @test only(logL) ≈ logL_bf
        end
    end

    @testset "Zero-probability observation before the last timestep" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.0, 1.0]), Categorical([0.5, 0.5])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        for obs_seq in ([1, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2, 2, 2])
            T = length(obs_seq)
            γ_bf, ξ_bf, _ = brute_force_posteriors(hsmm, obs_seq, fill(nothing, T), 2)
            storage = full_forward_backward(hsmm, obs_seq; seq_ends=(T,))
            @test all(isfinite, storage.γ)
            @test storage.γ ≈ γ_bf
            @test storage.ξ[1:(T - 1)] ≈ ξ_bf
            @test forward_backward(hsmm, obs_seq)[1] ≈ γ_bf
        end
        γ, _ = forward_backward(hsmm, [1, 2, 2])
        @test γ[:, 1] ≈ [0.0, 1.0]
    end

    @testset "Multiple sequences of differing lengths" begin
        rng = StableRNG(405)
        hsmm = rand_hsmm(rng, 3)
        lengths = (4, 11, 7, 1, 9, 5, 13)
        obs_seqs = [randn(rng, T) for T in lengths]
        obs_seq = reduce(vcat, obs_seqs)
        seq_ends = cumsum(collect(lengths))
        storage = full_forward_backward(hsmm, obs_seq; seq_ends)
        for k in eachindex(obs_seqs)
            t1, t2 = seq_limits(seq_ends, k)
            separate = full_forward_backward(
                hsmm, obs_seqs[k]; seq_ends=(lengths[k],), max_duration=maximum(lengths)
            )
            @test storage.γ[:, t1:t2] ≈ separate.γ
            @test storage.ξ[t1:t2] ≈ separate.ξ
            @test storage.logL[k] ≈ only(separate.logL)
        end
    end

    @testset "Empty subsequences" begin
        hsmm = rand_hsmm(StableRNG(5), 2)
        obs_seq = randn(StableRNG(6), 5)
        γ, logL = forward_backward(hsmm, obs_seq)
        γ_empty, logL_empty = forward_backward(hsmm, obs_seq; seq_ends=[0, 5, 5])
        @test γ_empty ≈ γ
        @test logL_empty ≈ [0.0, only(logL), 0.0]
    end

    @testset "Type promotion" begin
        hsmm32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4), Geometric(0.6)],
        )
        obs_seq = Float32[0.0, 5.0, 0.1, 4.9]
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_forward_backward(
            hsmm32, obs_seq, control_seq; seq_ends=(length(obs_seq),)
        )
        @test storage isa HSMMForwardBackwardStorage{Float64}
        @test eltype(storage.ξ) == Matrix{Float64}
        @test forward_backward(hsmm32, obs_seq)[1] isa Matrix{Float64}
    end

    @testset "error_if_not_finite" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        @test_throws ArgumentError forward_backward(hsmm, [1, 2, 3])
    end

    @testset "Allocations" begin
        rng = StableRNG(78)
        T = 30
        seq_ends = (T,)
        hsmm = rand_hsmm(rng, 3)
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        storage = initialize_forward_backward(hsmm, obs_seq, control_seq; seq_ends)
        call_forward_backward!(storage, hsmm, obs_seq, control_seq, seq_ends)  # warm up
        @test (@allocated call_forward_backward!(
            storage, hsmm, obs_seq, control_seq, seq_ends
        )) == 0
    end
end

# Unlike the marginals, maximization never sums probabilities, so `Float64` is precise enough.
function brute_force_viterbi(hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq))
    T = length(obs_seq)
    best_q, best_logL = zeros(Int, T), -Inf
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        logL = joint_logdensityof(hsmm, obs_seq, s, control_seq)
        if logL > best_logL
            best_q, best_logL = s, logL
        end
    end
    return best_q, best_logL
end

# Ensure `@allocated` sees concretely typed arguments.
function call_viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return viterbi!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

@testset "HSMM Viterbi" begin
    @testset "Brute-force maximization over all state sequences" begin
        rng = StableRNG(501)
        for N in 2:3, T in 5:7
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, control_seq, N)
            q, logL = viterbi(hsmm, obs_seq)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)
        end
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        hmm_init = [0.5, 0.3, 0.2]
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm_dists = [Normal(0.0), Normal(5.0), Normal(10.0)]
        hmm = HMM(hmm_init, hmm_trans, hmm_dists)

        N = length(hmm_init)
        hsmm_trans = [
            i == j ? 0.0 : hmm_trans[i, j] / (1 - hmm_trans[i, i]) for i in 1:N, j in 1:N
        ]
        hsmm_durs = [Geometric(1 - hmm_trans[i, i]) for i in 1:N]
        hsmm = HSMM(hmm_init, hsmm_trans, hmm_dists, hsmm_durs)

        # A long sequence also checks that the log-domain recursion does not lose precision.
        for (seed, T) in ((1, 15), (2, 15), (3, 500))
            obs_seq = rand(StableRNG(seed), hmm, T).obs_seq
            q_hmm, logL_hmm = viterbi(hmm, obs_seq)
            q, logL = viterbi(hsmm, obs_seq)
            @test q == q_hmm
            @test logL ≈ logL_hmm
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)
        end
    end

    @testset "Controlled duration distributions" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(4.0, 1.0)]
        durs = [[Geometric(0.15), Geometric(0.25)], [Geometric(0.85), Geometric(0.75)]]
        hsmm = ControlledDurationHSMM(init, trans, dists, durs)

        T = 6
        obs_seq = 2 .* randn(StableRNG(503), T) .+ 2
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], [2, 2, 2, 1, 1, 1])
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, control_seq, 2)
            q, logL = viterbi(hsmm, obs_seq, control_seq)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q, control_seq)
        end
    end

    @testset "Controlled transitions, durations and observations" begin
        hsmm = fully_controlled_hsmm()
        rng = StableRNG(507)
        T = 6
        obs_seq = 2 .* randn(rng, T) .+ 2
        for control_seq in ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], rand(rng, 1:2, T))
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, control_seq, 3)
            q, logL = viterbi(hsmm, obs_seq, control_seq)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q, control_seq)
        end

        # A `Fill` of a real control takes the fast path for constant controls.
        for c in 1:2
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, fill(c, T), 3)
            q, logL = viterbi(hsmm, obs_seq, Fill(c, T))
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test viterbi(hsmm, obs_seq, fill(c, T)) == (q, logL)
        end
    end

    @testset "Near-zero diagonal is ignored" begin
        # `valid_hsmm` tolerates diagonal entries up to `eps`.
        q_zero, logL_zero = viterbi(
            self_transition_hsmm([0.0 1.0; 1.0 0.0]), SELF_TRANSITION_OBS
        )
        # Staying in state 1 throughout would need the forbidden self-transition.
        @test q_zero != fill(1, length(SELF_TRANSITION_OBS))
        # `sparse` drops exact zeros but stores the near-zero diagonal.
        for trans in NEAR_ZERO_DIAGONALS
            q, logL = viterbi(self_transition_hsmm(trans), SELF_TRANSITION_OBS)
            @test q == q_zero
            @test logL ≈ logL_zero
        end
    end

    @testset "Sparse transition matrix" begin
        # Structural zeros must be read as impossible transitions, not as `log(1)`.
        init = [0.5, 0.3, 0.2]
        trans = [0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0]
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = [Geometric(0.3) for _ in 1:3]
        hsmm_dense = HSMM(init, trans, dists, durs)
        hsmm_sparse = HSMM(init, sparse(trans), dists, durs)
        # The direct path `2 → 3` is forbidden.
        obs_seq = [3.0, 3.0, 6.0, 6.0, 0.0, 3.0]
        T = length(obs_seq)
        control_seq = fill(nothing, T)
        q_bf, logL_bf = brute_force_viterbi(hsmm_sparse, obs_seq, control_seq, 3)
        q, logL = viterbi(hsmm_sparse, obs_seq)
        @test q == q_bf
        @test only(logL) ≈ logL_bf
        @test only(logL) ≈ joint_logdensityof(hsmm_sparse, obs_seq, q)
        @test viterbi(hsmm_dense, obs_seq) == (q, logL)

        seq_ends = (T,)
        storage = initialize_viterbi(hsmm_sparse, obs_seq, control_seq; seq_ends)
        call_viterbi!(storage, hsmm_sparse, obs_seq, control_seq, seq_ends)
        @test storage.q == q
        @test storage.logL == logL
        @test (@allocated call_viterbi!(
            storage, hsmm_sparse, obs_seq, control_seq, seq_ends
        )) == 0
    end

    @testset "Bounded duration support" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0), Normal(3.0)]
        hsmm = HSMM(init, trans, dists, [Binomial(2, 0.4), Binomial(1, 0.7)])
        rng = StableRNG(504)
        T = 7
        obs_seq = 3 .* randn(rng, T)
        q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, fill(nothing, T), 2)
        for max_duration in 3:T
            q, logL = viterbi(hsmm, obs_seq; max_duration)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)
        end

        # No state sequence fits the duration support.
        dists = [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])]
        hsmm = HSMM(init, trans, dists, [Binomial(1, 0.5), Binomial(1, 0.5)])
        @test_throws ArgumentError viterbi(hsmm, [1, 1, 1])
    end

    @testset "max_duration" begin
        rng = StableRNG(505)
        T = 7
        for N in 2:3, max_duration in 1:4
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, control_seq, N; max_duration)
            q, logL = viterbi(hsmm, obs_seq; max_duration)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            # Truncation restricts the search but does not bias the loglikelihood.
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)
            @test longest_segment(q) <= max_duration
        end

        # Long sojourns make truncation visible.
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        hsmm = HSMM(
            init, trans, [Normal(0.0), Normal(3.0)], [Geometric(0.1), Geometric(0.1)]
        )
        obs_seq = [zeros(6); fill(3.0, 2)]
        T = length(obs_seq)
        results = [viterbi(hsmm, obs_seq; max_duration) for max_duration in 1:T]
        logLs = [only(logL) for (_, logL) in results]
        @test issorted(logLs)
        for D in 2:T
            # The optimum only improves when it needs a sojourn of length `D` (ties can go either way).
            q = first(results[D])
            if !(logLs[D] ≈ logLs[D - 1])
                @test longest_segment(q) == D
            end
        end
        @test logLs[T] > logLs[1]
    end

    @testset "Zero-probability observations" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.0, 1.0]), Categorical([0.5, 0.5])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        for obs_seq in ([1, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2, 2, 2])
            T = length(obs_seq)
            q_bf, logL_bf = brute_force_viterbi(hsmm, obs_seq, fill(nothing, T), 2)
            q, logL = viterbi(hsmm, obs_seq)
            @test q == q_bf
            @test only(logL) ≈ logL_bf
            @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)
            # State 1 cannot emit observation 1.
            @test all(q[t] == 2 for t in 1:T if obs_seq[t] == 1)
        end

        dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8])]
        hsmm = HSMM(init, trans, dists, [Geometric(0.4), Geometric(0.6)])
        @test_throws ArgumentError viterbi(hsmm, [1, 2, 3])
    end

    @testset "Multiple sequences of differing lengths" begin
        rng = StableRNG(506)
        hsmm = rand_hsmm(rng, 3)
        lengths = (4, 11, 7, 1, 9, 5, 13)
        obs_seqs = [randn(rng, T) for T in lengths]
        obs_seq = reduce(vcat, obs_seqs)
        seq_ends = cumsum(collect(lengths))
        q, logL = viterbi(hsmm, obs_seq; seq_ends)
        for k in eachindex(obs_seqs)
            t1, t2 = seq_limits(seq_ends, k)
            q_k, logL_k = viterbi(hsmm, obs_seqs[k]; max_duration=maximum(lengths))
            @test q[t1:t2] == q_k
            @test logL[k] ≈ only(logL_k)
        end

        # The storage can be reused for other sequences of the same lengths.
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends)
        viterbi!(storage, hsmm, obs_seq, control_seq; seq_ends)
        @test storage.q == q
        @test storage.logL == logL
        other_obs_seq = randn(rng, length(obs_seq))
        viterbi!(storage, hsmm, other_obs_seq, control_seq; seq_ends)
        @test (storage.q, storage.logL) == viterbi(hsmm, other_obs_seq; seq_ends)
    end

    @testset "Empty subsequences" begin
        hsmm = rand_hsmm(StableRNG(5), 2)
        obs_seq = randn(StableRNG(6), 5)
        q, logL = viterbi(hsmm, obs_seq)
        q_empty, logL_empty = viterbi(hsmm, obs_seq; seq_ends=[0, 5, 5])
        @test q_empty == q
        @test logL_empty ≈ [0.0, only(logL), 0.0]
    end

    @testset "Type promotion" begin
        hsmm32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4), Geometric(0.6)],
        )
        obs_seq = Float32[0.0, 5.0, 0.1, 4.9]
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_viterbi(
            hsmm32, obs_seq, control_seq; seq_ends=(length(obs_seq),)
        )
        @test storage isa HSMMViterbiStorage{Float64}
        q, logL = viterbi(hsmm32, obs_seq)
        @test q == [1, 2, 1, 2]
        @test logL isa Vector{Float64}

        # Without any `Float64` component, the computation stays in `Float32`.
        hsmm32_only = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4f0), Geometric(0.6f0)],
        )
        q32, logL32 = viterbi(hsmm32_only, obs_seq)
        @test q32 == q
        @test logL32 isa Vector{Float32}
        @test logL32 ≈ logL
    end

    @testset "Allocations" begin
        rng = StableRNG(78)
        T = 30
        seq_ends = (T,)
        hsmm = rand_hsmm(rng, 3)
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        storage = initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends)
        call_viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends)  # warm up
        @test (storage.q, storage.logL) == viterbi(hsmm, obs_seq)
        @test (@allocated call_viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends)) == 0
    end
end

#= Expected numbers of completed and right-censored sojourns of each duration, by summing the
posterior weight of every state sequence over its segments. =#
function brute_force_duration_counts(
    hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq)
)
    T = length(obs_seq)
    D = min(max_duration, T)
    completed = zeros(BigFloat, D, N)
    censored = zeros(BigFloat, D, N)
    Z = big(0.0)
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        p = exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq)))
        Z += p
        for (t_start, t_end) in StateSegments(s, 1, T)
            counts = t_end == T ? censored : completed
            counts[t_end - t_start + 1, s[t_start]] += p
        end
    end
    return Float64.(completed ./ Z), Float64.(censored ./ Z)
end

function full_forward_backward_counts(
    hsmm, obs_seq, control_seq=fill(nothing, length(obs_seq)); kwargs...
)
    seq_ends = get(kwargs, :seq_ends, (length(obs_seq),))
    storage = initialize_forward_backward(
        hsmm, obs_seq, control_seq; duration_counts=true, kwargs...
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true)
    return storage
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward_backward!(
        storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
end

@testset "HSMM expected duration counts" begin
    @testset "Expected duration counts match brute force" begin
        rng = StableRNG(501)
        for N in 2:3, T in 5:6, max_duration in (2, T)
            hsmm = rand_hsmm(rng, N)
            obs_seq = randn(rng, T)
            control_seq = fill(nothing, T)
            completed_bf, censored_bf = brute_force_duration_counts(
                hsmm, obs_seq, control_seq, N; max_duration
            )
            storage = full_forward_backward_counts(
                hsmm, obs_seq; seq_ends=(T,), max_duration
            )
            @test storage.completed_durations ≈ completed_bf atol = 1e-12
            @test storage.censored_durations ≈ censored_bf atol = 1e-12
        end

        # The control at the start of a segment selects its duration distribution.
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(4.0, 1.0)]
        durs = [[Geometric(0.15), Geometric(0.25)], [Geometric(0.85), Geometric(0.75)]]
        hsmm = ControlledDurationHSMM(init, trans, dists, durs)
        T = 6
        obs_seq = randn(StableRNG(502), T)
        control_seq = [1, 1, 2, 2, 1, 1]
        completed_bf, censored_bf = brute_force_duration_counts(
            hsmm, obs_seq, control_seq, 2
        )
        storage = full_forward_backward_counts(hsmm, obs_seq, control_seq; seq_ends=(T,))
        @test storage.completed_durations ≈ completed_bf atol = 1e-12
        @test storage.censored_durations ≈ censored_bf atol = 1e-12
    end

    @testset "Expected duration counts are consistent with the marginals" begin
        rng = StableRNG(503)
        hsmm = rand_hsmm(rng, 3)
        lengths = (40, 1, 25, 60)
        obs_seq = reduce(vcat, [rand(rng, hsmm, T).obs_seq for T in lengths])
        seq_ends = cumsum(collect(lengths))
        storage = full_forward_backward_counts(hsmm, obs_seq; seq_ends)
        (; γ, ξ, completed_durations, censored_durations) = storage
        D = size(completed_durations, 1)
        @test D == maximum(lengths)
        # Each sequence ends with exactly one censored sojourn.
        @test sum(censored_durations) ≈ length(lengths)
        for j in 1:3
            # Every timestep spent in `j` belongs to exactly one sojourn.
            @test sum((1:D) .* (completed_durations[:, j] .+ censored_durations[:, j])) ≈
                sum(γ[j, :])
            # Every completed sojourn is followed by a transition out of `j`.
            @test sum(completed_durations[:, j]) ≈ sum(sum(ξₜ[j, :]) for ξₜ in ξ)
        end
        # The counts add up over sequences.
        for k in eachindex(lengths)
            t1, t2 = seq_limits(seq_ends, k)
            separate = full_forward_backward_counts(
                hsmm, obs_seq[t1:t2]; seq_ends=(t2 - t1 + 1,)
            )
            @test separate.completed_durations ≈ storage.seq_completed_durations[k]
            @test separate.censored_durations ≈ storage.seq_censored_durations[k]
        end
        # Counting does not change the marginals.
        storage_plain = full_forward_backward(hsmm, obs_seq; seq_ends)
        @test storage_plain.γ == γ
        @test storage_plain.ξ == ξ
    end

    @testset "Buffers are only allocated on request" begin
        rng = StableRNG(511)
        hsmm = rand_hsmm(rng, 2)
        T = 30
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        seq_ends = (T,)

        storage = initialize_forward_backward(hsmm, obs_seq, control_seq; seq_ends)
        @test size(storage.completed_durations, 1) == 0
        @test all(isempty, storage.seq_completed_durations)
        @test_throws ArgumentError forward_backward!(
            storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
    end

    @testset "Allocations" begin
        rng = StableRNG(512)
        lengths = (30,)
        seq_ends = lengths
        hsmm = rand_hsmm(rng, 3)
        obs_seq = rand(rng, hsmm, sum(lengths)).obs_seq
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)  # warm up
        @test (@allocated call_forward_backward_counts!(
            storage, hsmm, obs_seq, control_seq, seq_ends
        )) == 0
    end
end

function rand_sequences(rng, model, lengths)
    obs_seq = reduce(vcat, [rand(rng, model, T).obs_seq for T in lengths])
    return obs_seq, cumsum(lengths)
end

# Ensure `@allocated` sees concretely typed arguments.
function call_fit!(hsmm, storage, obs_seq, control_seq, seq_ends)
    return fit!(hsmm, storage, obs_seq, control_seq; seq_ends)
end

@testset "HSMM Baum-Welch" begin
    @testset "Recovery" begin
        init = [0.5, 0.3, 0.2]
        trans = [0.0 0.7 0.3; 0.4 0.0 0.6; 0.5 0.5 0.0]
        dists = [Normal(-3.0), Normal(0.0), Normal(3.0)]
        dur_dists = [Poisson(4.0), Poisson(1.0), Poisson(7.0)]
        obs_seq, seq_ends = rand_sequences(
            StableRNG(506), HSMM(init, trans, dists, dur_dists), fill(200, 50)
        )
        hsmm_guess = HSMM(
            fill(1 / 3, 3),
            [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0],
            [Normal(-2.0), Normal(0.5), Normal(2.0)],
            fill(Poisson(2.0), 3),
        )
        hsmm_est, logL_evolution = baum_welch(
            hsmm_guess, obs_seq; seq_ends, max_duration=30
        )
        # The guess must not be modified, even though the duration M-step writes `dists[i] = ...`.
        @test hsmm_guess.dur_dists == fill(Poisson(2.0), 3)
        @test all(>=(0), diff(logL_evolution))
        @test all(isapprox.(hsmm_est.init, init; atol=0.15))
        @test all(isapprox.(hsmm_est.trans, trans; atol=0.05))
        @test all(isapprox.(mean.(hsmm_est.dists), mean.(dists); atol=0.1))
        @test all(isapprox.(std.(hsmm_est.dists), std.(dists); atol=0.05))
        @test all(isapprox.(mean.(hsmm_est.dur_dists), mean.(dur_dists); rtol=0.08))
    end

    @testset "Short sequences" begin
        # Sequences of 2.5 mean sojourns, where leaving out the censored sojourns biases the durations.
        dists = [Normal(-2.0), Normal(2.0)]
        dur_dists = [Poisson(7.0), Poisson(4.0)]
        obs_seq, seq_ends = rand_sequences(
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
        hsmm_est, _ = baum_welch(hsmm_guess, obs_seq; seq_ends)
        @test all(isapprox.(mean.(hsmm_est.dur_dists), mean.(dur_dists); rtol=0.05))
        @test all(isapprox.(mean.(hsmm_est.dists), mean.(dists); atol=0.05))
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        hmm_trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
        hmm = HMM([0.5, 0.3, 0.2], hmm_trans, [Normal(0.0), Normal(5.0), Normal(10.0)])
        hsmm_from_hmm(hmm) = HSMM(
            hmm.init,
            [i == j ? 0.0 : hmm.trans[i, j] / (1 - hmm.trans[i, i]) for i in 1:3, j in 1:3],
            hmm.dists,
            [Geometric(1 - hmm.trans[i, i]) for i in 1:3],
        )
        # Run a fixed number of iterations, so that both runs stop at the same one.
        kwargs = (; max_iterations=10, atol=-Inf, loglikelihood_increasing=false)
        # Short sequences make the censored sojourns matter.
        for lengths in ([80, 120], fill(6, 30))
            obs_seq, seq_ends = rand_sequences(StableRNG(length(lengths)), hmm, lengths)
            hmm_est, logL_hmm = baum_welch(hmm, obs_seq; seq_ends, kwargs...)
            hsmm_est, logL_hsmm = baum_welch(
                hsmm_from_hmm(hmm), obs_seq; seq_ends, kwargs...
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

    hsmm_two = HSMM(
        [0.6, 0.4],
        [0.0 1.0; 1.0 0.0],
        [Normal(0.0), Normal(3.0)],
        [Geometric(0.3), Geometric(0.5)],
    )
    obs_seq_two, seq_ends_two = rand_sequences(StableRNG(507), hsmm_two, [30, 1, 45, 20])
    hsmm_vec, logL_vec = baum_welch(
        hsmm_two, obs_seq_two; seq_ends=seq_ends_two, max_iterations=10
    )

    @testset "Multiple sequences of differing lengths" begin
        # Vectors of sequence ends go through the threaded loops, tuples do not.
        hsmm_tup, logL_tup = baum_welch(
            hsmm_two, obs_seq_two; seq_ends=Tuple(seq_ends_two), max_iterations=10
        )
        @test logL_vec ≈ logL_tup
        @test hsmm_vec.trans ≈ hsmm_tup.trans
        @test succprob.(hsmm_vec.dur_dists) ≈ succprob.(hsmm_tup.dur_dists)
    end

    @testset "Empty subsequences" begin
        for seq_ends in ([0, 30, 30, 31, 76, 96], (30, 31, 31, 76, 96, 96))
            hsmm_empty, logL_empty = baum_welch(
                hsmm_two, obs_seq_two; seq_ends, max_iterations=10, max_duration=45
            )
            @test logL_empty ≈ logL_vec
            @test hsmm_empty.init ≈ hsmm_vec.init
            @test hsmm_empty.trans ≈ hsmm_vec.trans
            @test succprob.(hsmm_empty.dur_dists) ≈ succprob.(hsmm_vec.dur_dists)
        end
    end

    @testset "Sparse transition matrix" begin
        trans = sparse([0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0])
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = fill(Geometric(0.3), 3)
        obs_seq, seq_ends = rand_sequences(
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
        @test logL_evolution ≈ logL_dense
        @test hsmm_est.trans isa SparseMatrixCSC
        @test nnz(hsmm_est.trans) == nnz(hsmm_est.logtrans) == nnz(trans)
        @test hsmm_est.trans ≈ hsmm_dense.trans
        # Summing ξ must not drop the sparsity pattern of the scratch space.
        hsmm = deepcopy(hsmm_est)
        control_seq = Fill(nothing, length(obs_seq))
        storage = initialize_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)
        fit!(hsmm, storage, obs_seq; seq_ends)
        @test all(nnz(storage.ξ[t]) == nnz(trans) for t in seq_ends)
    end

    @testset "State never left" begin
        #= State `i` only emits `i`, and every sequence goes through states 1, 2 and 3 in this
        order, so all the sojourns of state 3 are censored. =#
        rng = StableRNG(522)
        obs_seq = Int[]
        seq_ends = Int[]
        for _ in 1:20
            append!(
                obs_seq,
                fill(1, rand(rng, 2:8)),
                fill(2, rand(rng, 2:8)),
                fill(3, rand(rng, 1:6)),
            )
            push!(seq_ends, length(obs_seq))
        end
        dists = [Categorical(Float64.(1:3 .== i)) for i in 1:3]
        trans_guess = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.3 0.7 0.0]
        for dur_dists in (Poisson.([2.0, 3.0, 4.0]), Geometric.([0.5, 0.3, 0.2]))
            hsmm_guess = HSMM(fill(1 / 3, 3), trans_guess, dists, dur_dists)
            hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends)
            @test all(>=(0), diff(logL_evolution))
            @test hsmm_est.init == [1.0, 0.0, 0.0]
            @test hsmm_est.trans[1:2, :] == [0.0 1.0 0.0; 0.0 0.0 1.0]
            @test hsmm_est.trans[3, :] == trans_guess[3, :]
            @test hsmm_est.dur_dists[1:2] != dur_dists[1:2]
            @test hsmm_est.dur_dists[3] == dur_dists[3]
        end
    end

    @testset "State never visited" begin
        # State 3 is neither initial nor reachable, so it gets no weight at all.
        trans = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.5 0.5 0.0]
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = fill(Geometric(0.3), 3)
        hsmm = HSMM([0.5, 0.5, 0.0], trans, dists, durs)
        obs_seq, seq_ends = rand_sequences(StableRNG(523), hsmm, fill(50, 4))
        hsmm_est, _ = baum_welch(hsmm, obs_seq; seq_ends)
        @test hsmm_est.init[3] == 0
        @test hsmm_est.trans[3, :] == trans[3, :]
        @test hsmm_est.dists[3] == dists[3]
        @test hsmm_est.dur_dists[3] == durs[3]
    end

    @testset "Type promotion" begin
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
        @test hsmm_est.trans isa Matrix{Float32}
        @test hsmm_est.dur_dists isa Vector{Geometric{Float64}}
    end

    @testset "Duration counts are required" begin
        hsmm = rand_hsmm(StableRNG(524), 3)
        obs_seq = rand(StableRNG(525), hsmm, 20).obs_seq
        seq_ends = (20,)
        storage = initialize_forward_backward(hsmm, obs_seq, Fill(nothing, 20); seq_ends)
        @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
    end

    @testset "Transition marginals are required" begin
        hsmm = rand_hsmm(StableRNG(527), 3)
        obs_seq = rand(StableRNG(528), hsmm, 20).obs_seq
        control_seq = Fill(nothing, 20)
        seq_ends = (20,)
        storage = initialize_forward_backward(
            hsmm,
            obs_seq,
            control_seq;
            seq_ends,
            transition_marginals=false,
            duration_counts=true,
        )
        forward_backward!(
            storage,
            hsmm,
            obs_seq,
            control_seq;
            seq_ends,
            transition_marginals=false,
            duration_counts=true,
        )
        @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
    end

    @testset "Sequences must cover the observations" begin
        # Otherwise the trailing columns of γ and ξ are uninitialized.
        hsmm = rand_hsmm(StableRNG(529), 3)
        obs_seq = rand(StableRNG(530), hsmm, 20).obs_seq
        control_seq = Fill(nothing, 20)
        seq_ends = (15,)
        storage = initialize_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        forward_backward!(
            storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
    end

    @testset "Trailing empty sequence" begin
        # It shares its end with the last nonempty sequence, whose ξ holds the total.
        hsmm = rand_hsmm(StableRNG(531), 3)
        obs_seq = rand(StableRNG(532), hsmm, 30).obs_seq
        hsmm_a, logL_a = baum_welch(hsmm, obs_seq; seq_ends=[10, 30], max_iterations=5)
        hsmm_b, logL_b = baum_welch(hsmm, obs_seq; seq_ends=[10, 30, 30], max_iterations=5)
        @test logL_a ≈ logL_b
        @test hsmm_a.trans ≈ hsmm_b.trans
        @test hsmm_a.init ≈ hsmm_b.init
    end

    @testset "Allocations" begin
        # Fitting a Distributions.jl `Normal` allocates, as in the HMM allocation tests.
        rng = StableRNG(526)
        (; init, trans, dur_dists) = rand_hsmm(rng, 3)
        dists = [LightCategorical([i == j ? 0.8 : 0.1 for j in 1:3]) for i in 1:3]
        hsmm = HSMM(init, trans, dists, dur_dists)
        T = 30
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        seq_ends = (T,)
        storage = initialize_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)
        call_fit!(hsmm, storage, obs_seq, control_seq, seq_ends)  # warm up
        call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)
        @test (@allocated call_fit!(hsmm, storage, obs_seq, control_seq, seq_ends)) == 0
    end
end

@testset "HSMM precision" begin
    # One huge negative `log b[2,1]` must not round away the later observations of state 2.
    init = [0.5, 0.5]
    trans = [0.0 1.0; 1.0 0.0]
    dists = [Normal(0.0, 1.0), Normal(0.0, 1e-7)]
    hsmm = HSMM(init, trans, dists, [Geometric(0.1), Geometric(0.1)])
    obs_seq = vcat(10.0, zeros(7))
    control_seq = fill(nothing, length(obs_seq))

    γ, _, logL = brute_force_posteriors(hsmm, obs_seq, control_seq, 2)
    @test logdensityof(hsmm, obs_seq) ≈ logL
    @test forward(hsmm, obs_seq)[1] ≈ brute_force_marginals(hsmm, obs_seq, control_seq, 2)
    γ_est, logL_est = forward_backward(hsmm, obs_seq)
    @test γ_est ≈ γ
    @test only(logL_est) ≈ logL
    q, best_logL = brute_force_viterbi(hsmm, obs_seq, control_seq, 2)
    q_est, best_logL_est = viterbi(hsmm, obs_seq)
    @test q_est == q
    @test only(best_logL_est) ≈ best_logL
end

@testset "HSMM type stability" begin
    rng = StableRNG(79)
    T = 20
    hsmm = rand_hsmm(rng, 3)
    obs_seq = rand(rng, hsmm, T).obs_seq
    control_seq = fill(nothing, T)
    seq_ends = (T,)
    forward_storage = initialize_forward(hsmm, obs_seq, control_seq; seq_ends)
    storage = initialize_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )

    @test_opt target_modules = (HiddenMarkovModels,) initialize_forward(
        hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) forward!(
        forward_storage, hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) forward(hsmm, obs_seq)
    @test_opt target_modules = (HiddenMarkovModels,) initialize_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) initialize_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
    @test_opt target_modules = (HiddenMarkovModels,) forward_backward!(
        storage, hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) forward_backward!(
        storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
    @test_opt target_modules = (HiddenMarkovModels,) forward_backward(hsmm, obs_seq)
    @test_opt target_modules = (HiddenMarkovModels,) logdensityof(hsmm, obs_seq)
    completed = [1.0, 2.0, 0.5]
    censored = [0.0, 0.3, 0.7]
    @test_opt target_modules = (HiddenMarkovModels,) fit_duration_in_sequence!(
        [Geometric(0.3)], 1, completed, censored
    )
    @test_opt target_modules = (HiddenMarkovModels,) fit_duration_in_sequence!(
        [Poisson(2.0)], 1, completed, censored
    )
    viterbi_storage = initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends)
    @test_opt target_modules = (HiddenMarkovModels,) initialize_viterbi(
        hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) viterbi!(
        viterbi_storage, hsmm, obs_seq, control_seq; seq_ends
    )
    @test_opt target_modules = (HiddenMarkovModels,) viterbi(hsmm, obs_seq)
    @test_opt target_modules = (HiddenMarkovModels,) baum_welch(
        hsmm, obs_seq, control_seq; seq_ends, max_duration=5
    )

    @test_call target_modules = (HiddenMarkovModels,) baum_welch(
        hsmm, obs_seq; seq_ends=[5, T]
    )
    @test_call target_modules = (HiddenMarkovModels,) forward_backward(
        hsmm, obs_seq; seq_ends=[5, T]
    )
    @test_call target_modules = (HiddenMarkovModels,) viterbi(
        hsmm, obs_seq; seq_ends=[5, T]
    )

    # Controls that vary take a separate code path.
    controlled_hsmm = fully_controlled_hsmm()
    int_control_seq = rand(rng, 1:2, T)
    @test_opt target_modules = (HiddenMarkovModels,) viterbi(
        controlled_hsmm, obs_seq, int_control_seq
    )
end
