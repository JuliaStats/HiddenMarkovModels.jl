two_states = ([0.6, 0.4], [0.0 1.0; 1.0 0.0])
sparse_trans = sparse([0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0])
sparse_hsmm = HSMM(
    [0.5, 0.3, 0.2],
    sparse_trans,
    [Normal(0.0), Normal(3.0), Normal(6.0)],
    fill(Geometric(0.3), 3),
)
# Sojourns last at most 3 timesteps.
bounded_hsmm = HSMM(
    two_states..., [Normal(0.0), Normal(3.0)], [Binomial(2, 0.4), Binomial(1, 0.7)]
)

@testset "Brute force" begin
    rng = StableRNG(63)
    scenarios = []
    for N in 2:3, max_duration in (1, 2, 3, 4, 7)
        push!(scenarios, ("Random", (rand_hsmm(rng, N), randn(rng, 7)), (; max_duration)))
    end
    obs_seq = 2 .* randn(rng, 6) .+ 2
    for control_seq in
        ([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 1, 1], rand(rng, 1:2, 6), Fill(2, 6))
        push!(scenarios, ("Controlled", (controlled_hsmm(), obs_seq, control_seq), (;)))
    end
    for obs_seq in (3 .* randn(rng, 6), [3.0, 3.0, 6.0, 6.0, 0.0, 3.0])
        push!(scenarios, ("Sparse", (sparse_hsmm, obs_seq), (;)))
    end
    for max_duration in (3, 7)
        push!(scenarios, ("Bounded", (bounded_hsmm, 3 .* randn(rng, 7)), (; max_duration)))
    end
    #= `valid_hsmm` tolerates diagonal entries up to `eps`, which must be ignored: a self-transition
    of `1e-17` in state 1 would explain `[1, 1, 1, 1]` better than the valid paths. =#
    for trans in ([1e-17 1.0; 1.0 1e-17], sparse([1e-17 1.0; 1.0 1e-17]))
        dists = [Categorical([1.0, 0.0]), Categorical([1e-20, 1 - 1e-20])]
        hsmm = HSMM([0.6, 0.4], trans, dists, fill(Binomial(1, 0.5), 2))
        push!(scenarios, ("Near-zero diagonal", (hsmm, [1, 1, 1, 1]), (;)))
    end
    # State 1 cannot emit `1`, even before the last timestep.
    dists = [Categorical([0.0, 1.0]), Categorical([0.5, 0.5])]
    hsmm = HSMM(two_states..., dists, [Geometric(0.4), Geometric(0.6)])
    for obs_seq in ([1, 2, 2], [2, 1, 2, 2], [2, 2, 1, 2, 2, 2])
        push!(scenarios, ("Impossible observation", (hsmm, obs_seq), (;)))
    end
    dur_dists = DuckDuration.([Geometric(0.4), Geometric(0.6)])
    hsmm = HSMM(two_states..., [Normal(0.0), Normal(5.0)], dur_dists)
    push!(scenarios, ("Duck-typed durations", (hsmm, randn(rng, 6)), (;)))
    # One huge negative `log b[2,1]` must not round away the later observations of state 2.
    hsmm = HSMM(two_states..., [Normal(0.0), Normal(0.0, 1e-7)], fill(Geometric(0.1), 2))
    push!(scenarios, ("Precision", (hsmm, vcat(10.0, zeros(7))), (;)))

    @testset "$name" for (name, args, kwargs) in scenarios
        test_brute_force(args...; kwargs...)
    end
    # Constant controls take a faster path.
    hsmm = controlled_hsmm()
    @test forward(hsmm, obs_seq, fill(2, 6)) == forward(hsmm, obs_seq, Fill(2, 6))
    @test viterbi(hsmm, obs_seq, fill(2, 6)) == viterbi(hsmm, obs_seq, Fill(2, 6))
end

@testset "Normalization" begin
    # The likelihoods of all observation sequences sum to 1, thanks to censoring.
    dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8]), Categorical([0.5, 0.5])]
    hsmm2 = HSMM(two_states..., dists[1:2], [Geometric(0.4), Geometric(0.55)])
    trans3 = [0.0 0.5 0.5; 0.4 0.0 0.6; 0.7 0.3 0.0]
    hsmm3 = HSMM([0.5, 0.3, 0.2], trans3, dists, Geometric.([0.3, 0.5, 0.9]))
    for hsmm in (hsmm2, hsmm3)
        obs_seqs = Iterators.product(ntuple(_ -> 1:2, 4)...)
        @test sum(exp(logdensityof(hsmm, collect(obs_seq))) for obs_seq in obs_seqs) ≈ 1
    end
end

@testset "HMM equivalence" begin
    rng = StableRNG(63)
    trans = [0.8 0.12 0.08; 0.1 0.85 0.05; 0.15 0.1 0.75]
    hmm = HMM([0.4, 0.35, 0.25], trans, [Normal(2.0), Normal(4.0), Normal(6.0)])
    hsmm = hsmm_from_hmm(hmm)
    # Observations near state 1 make long sojourns likely, which needs a precise survival.
    for obs_seq in (rand(rng, hmm, 15).obs_seq, rand(rng, hmm, 15).obs_seq, randn(rng, 500))
        T = length(obs_seq)
        state_seq = rand(rng, hmm, T).state_seq
        @test joint_logdensityof(hsmm, obs_seq, state_seq) ≈
            joint_logdensityof(hmm, obs_seq, state_seq)
        @test all(forward(hsmm, obs_seq) .≈ forward(hmm, obs_seq))
        @test all(forward_backward(hsmm, obs_seq) .≈ forward_backward(hmm, obs_seq))
        q, logL = viterbi(hsmm, obs_seq)
        q_hmm, logL_hmm = viterbi(hmm, obs_seq)
        @test q == q_hmm
        @test logL ≈ logL_hmm
        @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q)

        # The HMM self-transitions are the continued sojourns of the HSMM.
        ξ_hmm = forward_backward_storage(hmm, obs_seq).ξ
        ξ_hsmm = forward_backward_storage(hsmm, obs_seq).ξ
        @test all(ξ_hsmm .≈ [ξ - Diagonal(ξ) for ξ in ξ_hmm])
    end
end

@testset "Multiple sequences" begin
    rng = StableRNG(63)
    hsmm = rand_hsmm(rng, 3)
    obs_seqs = [randn(rng, T) for T in (4, 11, 1, 7, 13)]
    obs_seq, seq_ends = reduce(vcat, obs_seqs), cumsum(length.(obs_seqs))
    state_seqs = [rand(rng, 1:3, length(o)) for o in obs_seqs]
    control_seq = Fill(nothing, length(obs_seq))
    α, logL = forward(hsmm, obs_seq; seq_ends)
    γ, logL_fb = forward_backward(hsmm, obs_seq; seq_ends)
    q, logL_viterbi = viterbi(hsmm, obs_seq; seq_ends)
    storage = forward_backward_storage(hsmm, obs_seq; seq_ends, duration_counts=true)
    @test logL_fb ≈ logL
    @test γ ≈ storage.γ
    # Smoothing agrees with filtering at the end of each sequence.
    @test γ[:, seq_ends] ≈ α[:, seq_ends]
    @test joint_logdensityof(hsmm, obs_seq, reduce(vcat, state_seqs); seq_ends) ≈
        sum(joint_logdensityof.(Ref(hsmm), obs_seqs, state_seqs))
    for k in eachindex(obs_seqs, seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        storage_k = forward_backward_storage(hsmm, obs_seqs[k]; duration_counts=true)
        @test logL[k] ≈ logdensityof(hsmm, obs_seqs[k])
        @test α[:, t1:t2] ≈ first(forward(hsmm, obs_seqs[k]))
        @test storage.γ[:, t1:t2] ≈ storage_k.γ
        @test storage.ξ[t1:t2] ≈ storage_k.ξ
        @test storage.seq_completed_durations[k] ≈ storage_k.completed_durations
        @test storage.seq_censored_durations[k] ≈ storage_k.censored_durations
        q_k, logL_k = viterbi(hsmm, obs_seqs[k])
        @test q[t1:t2] == q_k
        @test logL_viterbi[k] ≈ only(logL_k)
    end

    # The storage can be reused for other sequences of the same lengths.
    viterbi_storage = initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends)
    for obs_seq in (obs_seq, randn(rng, length(obs_seq)))
        viterbi!(viterbi_storage, hsmm, obs_seq, control_seq; seq_ends)
        @test (viterbi_storage.q, viterbi_storage.logL) == viterbi(hsmm, obs_seq; seq_ends)
    end
end

@testset "Truncation" begin
    rng = StableRNG(63)
    # Long sojourns make truncation visible, and distinct observations avoid ties in Viterbi.
    hsmm = HSMM(
        [0.5, 0.5], two_states[2], [Normal(0.0), Normal(3.0)], fill(Geometric(0.1), 2)
    )
    obs_seq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.0]
    T = length(obs_seq)
    logLs = [logdensityof(hsmm, obs_seq; max_duration) for max_duration in 1:(T + 3)]
    @test issorted(logLs)
    @test all(logLs[1:(T - 1)] .< logdensityof(hsmm, obs_seq))
    @test all(logLs[T:end] .== logdensityof(hsmm, obs_seq))
    results = [viterbi(hsmm, obs_seq; max_duration) for max_duration in 1:T]
    logLs = [only(logL) for (_, logL) in results]
    @test issorted(logLs)
    for D in 2:T
        # The optimum improves exactly when it needs a sojourn of length `D`.
        @test (logLs[D] > logLs[D - 1]) == (longest_segment(first(results[D])) == D)
    end
    @test logLs[T] > logLs[1]

    # Beyond the duration support, truncation changes nothing.
    obs_seq = 3 .* randn(rng, 7)
    exact = logdensityof(bounded_hsmm, obs_seq)
    @test all(logdensityof(bounded_hsmm, obs_seq; max_duration=D) == exact for D in 3:7)
    @test logdensityof(bounded_hsmm, obs_seq; max_duration=2) < exact
end

@testset "Long sequences" begin
    rng = StableRNG(63)
    hsmm = rand_hsmm(rng, 3)
    obs_seq, seq_ends = rand_sequences(rng, hsmm, [40, 1, 25, 60])
    storage = forward_backward_storage(hsmm, obs_seq; seq_ends, duration_counts=true)
    (; γ, ξ, completed_durations, censored_durations) = storage
    @test all(>=(0), γ)
    @test all(sum(γ; dims=1) .≈ 1)
    @test all(sum(storage.forward.α; dims=1) .≈ 1)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # The mass of a state changes only through segment boundaries.
        for t in t1:(t2 - 1)
            inflow, outflow = vec(sum(ξ[t]; dims=1)), vec(sum(ξ[t]; dims=2))
            @test γ[:, t + 1] - γ[:, t] ≈ inflow - outflow atol = 1e-12
        end
    end
    @test size(completed_durations, 1) == 60
    # Each sequence ends with exactly one censored sojourn.
    @test sum(censored_durations) ≈ length(seq_ends)
    D = 1:size(completed_durations, 1)
    for j in 1:3
        # Every timestep spent in `j` belongs to exactly one sojourn.
        @test sum(D .* (completed_durations[:, j] .+ censored_durations[:, j])) ≈
            sum(γ[j, :])
        # Every completed sojourn is followed by a transition out of `j`.
        @test sum(completed_durations[:, j]) ≈ sum(sum(ξₜ[j, :]) for ξₜ in ξ)
    end

    # Counting does not change the marginals.
    storage_plain = forward_backward_storage(hsmm, obs_seq; seq_ends)
    @test storage_plain.γ == γ
    @test storage_plain.ξ == ξ
end

@testset "Sparse" begin
    rng = StableRNG(63)
    (; init, dists, dur_dists) = sparse_hsmm
    hsmm_dense = HSMM(init, Matrix(sparse_trans), dists, dur_dists)
    obs_seq = 3 .* randn(rng, 10)
    @test all(forward(sparse_hsmm, obs_seq) .≈ forward(hsmm_dense, obs_seq))
    @test viterbi(sparse_hsmm, obs_seq) == viterbi(hsmm_dense, obs_seq)
    storages = [forward_backward_storage(h, obs_seq) for h in (sparse_hsmm, hsmm_dense)]
    @test storages[1].γ ≈ storages[2].γ
    @test storages[1].ξ ≈ storages[2].ξ
    # The forbidden transitions keep their structural zeros.
    @test all(
        ξₜ -> ξₜ isa SparseMatrixCSC && nnz(ξₜ) == nnz(sparse_trans), storages[1].ξ[1:9]
    )

    # Structural zeros are impossible transitions, not `log(1)`.
    @test joint_logdensityof(sparse_hsmm, [3.0, 6.0, 0.0], [2, 3, 1]) == -Inf
    @test joint_logdensityof(sparse_hsmm, [3.0, 6.0, 0.0], [1, 2, 1]) ==
        joint_logdensityof(hsmm_dense, [3.0, 6.0, 0.0], [1, 2, 1])
end

@testset "Type promotion" begin
    obs_seq = Float32[0.0, 5.0, 0.1, 4.9]
    control_seq, seq_ends = Fill(nothing, 4), (4,)
    dists32 = [Normal(0.0f0), Normal(5.0f0)]
    # Float64 durations promote an otherwise Float32 model.
    init32, trans32 = Float32.(two_states[1]), Float32.(two_states[2])
    hsmms = [
        HSMM(init32, trans32, dists32, Geometric.(R[0.4, 0.6])) for R in (Float64, Float32)
    ]
    for (hsmm, R) in zip(hsmms, (Float64, Float32))
        @test initialize_forward(hsmm, obs_seq, control_seq; seq_ends) isa
            HSMMForwardStorage{R}
        storage = initialize_forward_backward(hsmm, obs_seq, control_seq; seq_ends)
        @test storage isa HSMMForwardBackwardStorage{R}
        @test eltype(storage.ξ) == Matrix{R}
        @test initialize_viterbi(hsmm, obs_seq, control_seq; seq_ends) isa
            HSMMViterbiStorage{R}
        @test logdensityof(hsmm, obs_seq) isa R
        @test joint_logdensityof(hsmm, obs_seq, [1, 2, 1, 2]) isa R
        @test first(forward_backward(hsmm, obs_seq)) isa Matrix{R}
        @test viterbi(hsmm, obs_seq) isa Tuple{Vector{Int},Vector{R}}
    end
    q, logL = viterbi(hsmms[1], obs_seq)
    @test q == [1, 2, 1, 2]
    @test viterbi(hsmms[2], obs_seq)[2] ≈ logL
end

@testset "Errors" begin
    rng = StableRNG(63)
    hsmm = rand_hsmm(rng, 2)
    obs_seq, state_seq = randn(rng, 5), [1, 1, 2, 2, 1]
    @test_throws ArgumentError logdensityof(hsmm, obs_seq; max_duration=0)
    # Empty sequences.
    for seq_ends in ([0, 5], [2, 2, 5], (5, 5))
        @test_throws ArgumentError forward(hsmm, obs_seq; seq_ends)
        @test_throws ArgumentError viterbi(hsmm, obs_seq; seq_ends)
        @test_throws ArgumentError forward_backward(hsmm, obs_seq; seq_ends)
        @test_throws ArgumentError logdensityof(hsmm, obs_seq; seq_ends)
        @test_throws ArgumentError joint_logdensityof(hsmm, obs_seq, state_seq; seq_ends)
        @test_throws ArgumentError baum_welch(hsmm, obs_seq; seq_ends)
    end

    # The observation `3` is outside the support of every observation distribution.
    dists = [Categorical([0.7, 0.3]), Categorical([0.2, 0.8])]
    hsmm = HSMM(two_states..., dists, [Geometric(0.4), Geometric(0.6)])
    @test logdensityof(hsmm, [1, 2, 3]) == -Inf
    @test_throws ArgumentError forward(hsmm, [1, 2, 3])
    @test_throws ArgumentError forward(hsmm, [1, 2, 3]; error_if_not_finite=true)
    @test only(last(forward(hsmm, [1, 2, 3]; error_if_not_finite=false))) == -Inf
    @test_throws ArgumentError forward_backward(hsmm, [1, 2, 3])
    @test_throws ArgumentError viterbi(hsmm, [1, 2, 3])

    # No state sequence fits the duration support.
    dists = [Categorical([1.0, 0.0]), Categorical([0.0, 1.0])]
    hsmm = HSMM(two_states..., dists, fill(Binomial(1, 0.5), 2))
    @test_throws ArgumentError viterbi(hsmm, [1, 1, 1])
end

@testset "Autodiff" begin
    rng = StableRNG(63)
    # The gradients must be finite, which `≈` checks.
    fd_gradient(f, θ) = only(FiniteDifferences.grad(central_fdm(5, 1), f, θ))
    trans3 = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0]

    # Each model makes `-Inf` terms reach the log-domain recursion.
    function zero_off_diagonal(θ)
        trans = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
        dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
        return HSMM(fill(1 / 3, 3), trans, dists, [Geometric(θ[3]) for _ in 1:3])
    end
    function zero_init(θ)
        init = [one(θ[1]), zero(θ[1]), zero(θ[1])]
        dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
        return HSMM(init, trans3 .* θ[3] ./ θ[3], dists, [Geometric(0.3) for _ in 1:3])
    end
    function bounded_durations(θ)
        dists = [Normal(θ[1]), Normal(θ[2]), Normal(6.0)]
        return HSMM(fill(1 / 3, 3), trans3, dists, [Binomial(4, θ[3]) for _ in 1:3])
    end
    obs_seq = randn(rng, 20) .* 2 .+ 3
    θ = [0.0, 3.0, 0.4]
    for build in (zero_off_diagonal, zero_init, bounded_durations)
        f(θ) = logdensityof(build(θ), obs_seq)
        @test ForwardDiff.gradient(f, θ) ≈ fd_gradient(f, θ) rtol = 1e-5
    end

    # `b[1,t] = 0` is a `Dual` whose log has NaN partials, which must not leak into the gradient.
    function impossible_obs(θ)
        dists = [Categorical([zero(θ[1]), θ[1], 1 - θ[1]]), Categorical([0.3, 0.3, 0.4])]
        return HSMM([0.5, 0.5], two_states[2], dists, [Geometric(θ[2]), Geometric(0.5)])
    end
    f_obs(θ) = logdensityof(impossible_obs(θ), [1, 2, 3, 3, 1, 2])
    @test ForwardDiff.gradient(f_obs, [0.4, 0.3]) ≈ fd_gradient(f_obs, [0.4, 0.3]) rtol =
        1e-5

    # Zygote uses a custom rule.
    function build(θ)
        init = [θ[1], θ[2], 1 - θ[1] - θ[2]]
        trans = [0 θ[3] 1-θ[3]; θ[4] 0 1-θ[4]; θ[5] 1-θ[5] 0]
        dists = [Normal(θ[6]), Normal(θ[7]), Normal(θ[8])]
        return HSMM(init, trans, dists, Geometric.(θ[9:11]))
    end
    θ = [0.5, 0.3, 0.6, 0.4, 0.7, -2.0, 0.0, 2.0, 0.3, 0.2, 0.5]
    obs_seq = rand(rng, build(θ), 60).obs_seq
    seq_ends = [20, 21, 45, 60]
    for kwargs in ((;), (; seq_ends), (; seq_ends, max_duration=5))
        f(θ) = logdensityof(build(θ), obs_seq; kwargs...)
        @test only(Zygote.gradient(f, θ)) ≈ ForwardDiff.gradient(f, θ)
    end
    # At a zero initial probability, the derivative is one-sided.
    θ0 = [θ[1]; 0.0; θ[3:end]]
    f0(θ) = logdensityof(build(θ), obs_seq)
    h = 1e-7
    @test only(Zygote.gradient(f0, θ0))[2] ≈ (f0(θ0 + h * (1:11 .== 2)) - f0(θ0)) / h rtol =
        1e-5
    # The duration counts are summed over all start times.
    @test_throws ArgumentError Zygote.gradient(
        θ -> logdensityof(build(θ), obs_seq, collect(1:60)), θ
    )
end

# The fixtures of `correctness.jl` are uncontrolled and dense, and skip the duration counts.
@testset "Type stability" begin
    rng = StableRNG(63)
    test_type_stability(rng, controlled_hsmm(), rand(rng, 1:2, 20); seq_ends=[5, 20])
end

# `@allocated` needs concretely typed arguments.
function fb_allocations(storage, hsmm, obs_seq, control_seq, seq_ends)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return @allocated forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

@testset "Allocations" begin
    rng = StableRNG(63)
    control_seq, seq_ends = Fill(nothing, 30), (30,)
    test_allocations(rng, sparse_hsmm, control_seq; seq_ends)
    for hsmm in (rand_hsmm(rng, 3), sparse_hsmm)
        obs_seq = rand(rng, hsmm, 30).obs_seq
        storage = initialize_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
        )
        @test fb_allocations(storage, hsmm, obs_seq, control_seq, seq_ends) == 0
    end
end
