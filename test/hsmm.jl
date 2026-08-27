using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using HiddenMarkovModels
using HiddenMarkovModels:
    AbstractLatentStateModel,
    AbstractHSMM,
    HSMMForwardStorage,
    duration_distributions,
    duration_logdensityof,
    duration_logsurvival,
    duration_logdensity_type,
    elementwise_log,
    StateSegments,
    forward!,
    initialize_hsmm_forward,
    log_initialization,
    log_transition_matrix,
    valid_hsmm
using Distributions: Categorical, Geometric, Normal
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

all_state_seqs(N, T) = (collect(s) for s in Iterators.product(ntuple(_ -> 1:N, T)...))

# An independent oracle for `forward`.
function brute_force_logdensityof(hsmm, obs_seq, control_seq, N)
    T = length(obs_seq)
    return log(
        sum(
            exp(joint_logdensityof(hsmm, obs_seq, s, control_seq)) for
            s in all_state_seqs(N, T)
        ),
    )
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
        # Repeat to expose thread scheduling nondeterminism.
        for _ in 1:20
            @test logdensityof(hsmm, obs_seq; seq_ends) == joint
        end
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
        end
        @test logdensityof(hsmm, obs_seq, [1, 1, 2, 2, 1, 1]) !=
            logdensityof(hsmm, obs_seq, [2, 2, 1, 1, 2, 2])
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

    @testset "Type promotion" begin
        hsmm32 = HSMM(
            Float32[0.6, 0.4],
            Float32[0.0 1.0; 1.0 0.0],
            [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
            [Geometric(0.4), Geometric(0.6)],
        )
        obs_seq = Float32[0.0, 5.0, 0.1, 4.9]
        control_seq = fill(nothing, length(obs_seq))
        storage = initialize_hsmm_forward(
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
        storage32 = initialize_hsmm_forward(
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

        storage = initialize_hsmm_forward(
            hsmm, obs_seq, fill(nothing, T); seq_ends=(T,), max_duration=T
        )
        forward!(storage, hsmm, obs_seq, fill(nothing, T); seq_ends=(T,))
        for t in 1:T
            @test storage.log_prefix[t] ≈ logdensityof(hsmm, obs_seq[1:t])
            @test storage.α[:, t] ≈ forward(hsmm, obs_seq[1:t])[1][:, t]
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

    @testset "Allocations" begin
        rng = StableRNG(77)
        hsmm = rand_hsmm(rng, 3)
        T = 30
        obs_seq = rand(rng, hsmm, T).obs_seq
        control_seq = fill(nothing, T)
        # An `NTuple{1}` for `seq_ends` disables multithreading, as in the HMM allocation test.
        seq_ends = (T,)
        storage = initialize_hsmm_forward(hsmm, obs_seq, control_seq; seq_ends)
        call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)  # warm up
        @test (@allocated call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)) == 0
    end
end
