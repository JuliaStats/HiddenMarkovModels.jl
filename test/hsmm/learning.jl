@testset "HMM equivalence" begin
    rng = StableRNG(63)
    trans = [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5]
    hmm = HMM([0.5, 0.3, 0.2], trans, [Normal(0.0), Normal(5.0), Normal(10.0)])
    # Run a fixed number of iterations, so that both runs stop at the same one.
    kwargs = (; max_iterations=10, atol=-Inf, loglikelihood_increasing=false)
    # Short sequences make the censored sojourns matter.
    for lengths in ([80, 120], fill(6, 30))
        obs_seq, seq_ends = rand_sequences(rng, hmm, lengths)
        hmm_est, logL_hmm = baum_welch(hmm, obs_seq; seq_ends, kwargs...)
        hsmm_est, logL_hsmm = baum_welch(hsmm_from_hmm(hmm), obs_seq; seq_ends, kwargs...)
        hsmm_ref = hsmm_from_hmm(hmm_est)
        @test logL_hsmm ≈ logL_hmm
        @test initialization(hsmm_est) ≈ initialization(hsmm_ref)
        @test transition_matrix(hsmm_est) ≈ transition_matrix(hsmm_ref)
        @test succprob.(duration_distributions(hsmm_est)) ≈
            succprob.(duration_distributions(hsmm_ref))
        @test mean.(obs_distributions(hsmm_est)) ≈ mean.(obs_distributions(hsmm_ref))
        @test std.(obs_distributions(hsmm_est)) ≈ std.(obs_distributions(hsmm_ref))
    end
    @test_throws ArgumentError baum_welch(hmm, randn(rng, 10); max_duration=5)
end

@testset "Censoring" begin
    # Sequences of 2.5 mean sojourns, where leaving out the censored sojourns biases the durations.
    rng = StableRNG(63)
    trans = [0.0 1.0; 1.0 0.0]
    dists = [Normal(-2.0), Normal(2.0)]
    dur_dists = [Poisson(7.0), Poisson(4.0)]
    obs_seq, seq_ends = rand_sequences(
        rng, HSMM([0.5, 0.5], trans, dists, dur_dists), fill(20, 200)
    )
    dists_guess = [Normal(-1.0, 1.5), Normal(1.0, 1.5)]
    hsmm_guess = HSMM([0.6, 0.4], trans, dists_guess, [Poisson(3.0), Poisson(10.0)])
    hsmm_est, _ = baum_welch(hsmm_guess, obs_seq; seq_ends)
    @test all(
        isapprox.(mean.(duration_distributions(hsmm_est)), mean.(dur_dists); rtol=0.05)
    )
    @test all(isapprox.(mean.(obs_distributions(hsmm_est)), mean.(dists); atol=0.05))
end

@testset "Threading" begin
    # Vectors of sequence ends go through the threaded loops, tuples do not.
    rng = StableRNG(63)
    hsmm = HSMM(
        [0.6, 0.4], [0.0 1.0; 1.0 0.0], [Normal(0.0), Normal(3.0)], Geometric.([0.3, 0.5])
    )
    obs_seq, seq_ends = rand_sequences(rng, hsmm, [30, 1, 45, 20])
    results = map((seq_ends, Tuple(seq_ends))) do seq_ends
        return baum_welch(hsmm, obs_seq; seq_ends, max_iterations=10, max_duration=25)
    end
    (hsmm_vec, logL_vec), (hsmm_tup, logL_tup) = results
    @test logL_vec ≈ logL_tup
    @test transition_matrix(hsmm_vec) ≈ transition_matrix(hsmm_tup)
    @test succprob.(duration_distributions(hsmm_vec)) ≈
        succprob.(duration_distributions(hsmm_tup))
end

@testset "Sparse" begin
    rng = StableRNG(63)
    trans = sparse([0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0])
    dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
    hsmm = HSMM([0.5, 0.3, 0.2], trans, dists, fill(Geometric(0.3), 3))
    obs_seq, seq_ends = rand_sequences(rng, hsmm, fill(100, 3))
    trans_guess = sparse([0.0 0.3 0.7; 1.0 0.0 0.0; 1.0 0.0 0.0])
    dists_guess = [Normal(0.5), Normal(2.5), Normal(5.0)]
    hsmm_guess, hsmm_dense_guess = map((trans_guess, Matrix(trans_guess))) do trans
        return HSMM([0.5, 0.3, 0.2], trans, dists_guess, fill(Geometric(0.5), 3))
    end
    hsmm_est, logL_evolution = baum_welch(hsmm_guess, obs_seq; seq_ends)
    hsmm_dense, logL_dense = baum_welch(hsmm_dense_guess, obs_seq; seq_ends)
    @test logL_evolution ≈ logL_dense
    @test transition_matrix(hsmm_est) isa SparseMatrixCSC
    @test nnz(transition_matrix(hsmm_est)) == nnz(trans)
    @test nnz(log_transition_matrix(hsmm_est)) == nnz(trans)
    @test transition_matrix(hsmm_est) ≈ transition_matrix(hsmm_dense)
    # Summing ξ must not drop the sparsity pattern of the scratch space.
    storage = forward_backward_storage(hsmm_est, obs_seq; seq_ends, duration_counts=true)
    fit!(deepcopy(hsmm_est), storage, obs_seq; seq_ends)
    @test all(nnz(storage.ξ[t]) == nnz(trans) for t in seq_ends)
end

@testset "Unvisited states" begin
    rng = StableRNG(63)
    #= State `i` only emits `i`, and every sequence goes through states 1, 2 and 3 in this order,
    so all the sojourns of state 3 are censored and it is never left. =#
    obs_seq, seq_ends = Int[], Int[]
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
        trans_est = transition_matrix(hsmm_est)
        dur_dists_est = duration_distributions(hsmm_est)
        @test all(>=(0), diff(logL_evolution))
        @test initialization(hsmm_est) == [1.0, 0.0, 0.0]
        @test trans_est[1:2, :] == [0.0 1.0 0.0; 0.0 0.0 1.0]
        @test trans_est[3, :] == trans_guess[3, :]
        @test dur_dists_est[1:2] != dur_dists[1:2]
        @test dur_dists_est[3] == dur_dists[3]
    end

    # State 3 is neither initial nor reachable, so it gets no weight at all.
    trans = [0.0 1.0 0.0; 1.0 0.0 0.0; 0.5 0.5 0.0]
    dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
    hsmm = HSMM([0.5, 0.5, 0.0], trans, dists, fill(Geometric(0.3), 3))
    obs_seq, seq_ends = rand_sequences(rng, hsmm, fill(50, 4))
    hsmm_est, _ = baum_welch(hsmm, obs_seq; seq_ends)
    @test initialization(hsmm_est)[3] == 0
    @test transition_matrix(hsmm_est)[3, :] == trans[3, :]
    @test obs_distributions(hsmm_est)[3] == dists[3]
    @test duration_distributions(hsmm_est)[3] == Geometric(0.3)
end

@testset "Type promotion" begin
    # Float32 observations and Float64 durations promote the storage to Float64.
    hsmm = HSMM(
        Float32[0.6, 0.4],
        Float32[0.0 1.0; 1.0 0.0],
        [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)],
        [Geometric(0.4), Geometric(0.6)],
    )
    hsmm_est, logL_evolution = baum_welch(hsmm, rand(StableRNG(63), hsmm, 200).obs_seq)
    @test logL_evolution isa Vector{Float64}
    @test transition_matrix(hsmm_est) isa Matrix{Float32}
    @test duration_distributions(hsmm_est) isa Vector{Geometric{Float64}}
end

@testset "Storage requirements" begin
    rng = StableRNG(63)
    hsmm = rand_hsmm(rng, 3)
    obs_seq = rand(rng, hsmm, 20).obs_seq
    control_seq, seq_ends = Fill(nothing, 20), (20,)
    function init_fb(seq_ends=seq_ends; kwargs...)
        return initialize_forward_backward(hsmm, obs_seq, control_seq; seq_ends, kwargs...)
    end

    # Duration counts are required.
    @test_throws ArgumentError fit!(hsmm, init_fb(), obs_seq; seq_ends)
    @test_throws ArgumentError forward_backward!(
        init_fb(), hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
    # With room for them, `forward_backward!` fills them by default, like `baum_welch`.
    storage = init_fb(; duration_counts=true)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    hsmm_est = deepcopy(hsmm)
    fit!(hsmm_est, storage, obs_seq; seq_ends)
    hsmm_bw, _ = baum_welch(hsmm, obs_seq; max_iterations=1)
    @test succprob.(duration_distributions(hsmm_est)) ≈
        succprob.(duration_distributions(hsmm_bw))

    # Transition marginals are required.
    storage = init_fb(; transition_marginals=false, duration_counts=true)
    @test_throws ArgumentError forward_backward!(
        storage, hsmm, obs_seq, control_seq; seq_ends, transition_marginals=true
    )
    # Without room for them, `forward_backward!` skips them by default.
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)

    # Sequences must cover the observations, otherwise the end of γ and ξ is uninitialized.
    storage = init_fb((15,); duration_counts=true)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends=(15,))
    @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends=(15,))
end
