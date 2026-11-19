init = [0.6, 0.4]
trans = [0.0 1.0; 1.0 0.0]
dists = [Normal(0.0), Normal(5.0)]
dur_dists = [Geometric(0.4), Geometric(0.6)]

@testset "Constructor" begin
    hsmm = HSMM(init, trans, dists, dur_dists)
    @test length(hsmm) == 2
    @test initialization(hsmm) == init
    @test transition_matrix(hsmm) == trans
    @test obs_distributions(hsmm) === dists
    @test duration_distributions(hsmm) === dur_dists
    @test valid_hsmm(hsmm)
    @test log_initialization(hsmm) == elementwise_log(init)
    @test log_transition_matrix(hsmm) == elementwise_log(trans)
    @test DensityKind(hsmm) == HasDensity()
    @test HSMM(init, [0 1; 1 0], dists, dur_dists) isa HSMM
    # Mixing number types within a family is fine.
    @test HSMM(init, trans, dists, Geometric[Geometric(0.4), Geometric(0.6f0)]) isa HSMM

    @test_throws ArgumentError HSMM(init, [0.1 0.9; 1.0 0.0], dists, dur_dists)
    @test_throws ArgumentError HSMM(init, trans, dists, dur_dists[1:1])
    @test_throws ArgumentError HSMM([1.0], trans, dists, dur_dists)
    @test_throws ArgumentError HSMM(init, trans, dists, [Geometric(0.4), Poisson(2.0)])
end

@testset "Sampling" begin
    rng = StableRNG(63)
    trans3 = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0]
    dists3 = [Normal(0.0), Normal(5.0), Normal(10.0)]
    hsmm = HSMM([1.0, 0.0, 0.0], trans3, dists3, fill(Geometric(0.3), 3))
    (; state_seq, obs_seq, duration_seq) = rand(rng, hsmm, 200)
    @test length(state_seq) == length(obs_seq) == length(duration_seq) == 200
    @test all(in(1:3), state_seq)
    @test all(>=(1), duration_seq)
    @test state_seq[1] == 1
    @test eltype(obs_seq) == Float64

    # Sojourns of length 1 and a zero diagonal forbid consecutive repeats.
    hsmm = HSMM([1.0, 0.0, 0.0], trans3, dists3, fill(Geometric(1.0), 3))
    (; state_seq, duration_seq) = rand(rng, hsmm, 100)
    @test all(==(1), duration_seq)
    @test all(state_seq[1:(end - 1)] .!= state_seq[2:end])
end

@testset "Type hierarchy" begin
    hmm = HMM([0.5, 0.5], [0.7 0.3; 0.2 0.8], dists)
    @test AbstractHMM <: AbstractLatentStateModel
    @test AbstractHSMM <: AbstractLatentStateModel
    @test !(AbstractHMM <: AbstractHSMM)
    @test !(AbstractHSMM <: AbstractHMM)
    @test HSMM <: AbstractHSMM
    @test !hasmethod(duration_distributions, Tuple{typeof(hmm)})
    # HMM sampling keeps its 2-tuple.
    @test propertynames(rand(StableRNG(63), hmm, 50)) == (:state_seq, :obs_seq)
end

@testset "Eltype" begin
    dists32 = [Normal(0.0f0), Normal(5.0f0)]
    @test duration_logdensity_type(HSMM(init, trans, dists, dur_dists), nothing) === Float64
    hsmm32 = HSMM(init, trans, dists, Geometric.(Float32[0.4, 0.6]))
    @test duration_logdensity_type(hsmm32, nothing) === Float32
    # Float64 durations promote an otherwise Float32 model.
    hsmm = HSMM(Float32.(init), Float32.(trans), dists32, dur_dists)
    @test eltype(hsmm, 0.0f0, nothing) === Float64
    # HMMs ignore the duration machinery.
    hmm32 = HMM(Float32[0.5, 0.5], Float32[0.7 0.3; 0.2 0.8], dists32)
    @test !hasmethod(duration_logdensity_type, Tuple{typeof(hmm32),Nothing})
    @test eltype(hmm32, 0.0f0, nothing) === Float32
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

@testset "Joint loglikelihood" begin
    hsmm = HSMM(init, trans, dists, dur_dists)
    # A completed sojourn of state 1 lasting 2 steps, then a censored one of state 2.
    obs_seq = [0.3, -0.1, 5.2]
    expected =
        log(init[1]) +
        duration_logdensityof(dur_dists[1], 2) +
        log(trans[1, 2]) +
        duration_logsurvival(dur_dists[2], 1) +
        sum(logdensityof.(dists[[1, 1, 2]], obs_seq))
    @test joint_logdensityof(hsmm, obs_seq, [1, 1, 2]) ≈ expected
end
