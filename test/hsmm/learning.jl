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
        @test duration_distributions(hsmm_guess) == fill(Poisson(2.0), 3)
        @test all(>=(0), diff(logL_evolution))
        @test all(isapprox.(initialization(hsmm_est), init; atol=0.15))
        @test all(isapprox.(transition_matrix(hsmm_est), trans; atol=0.05))
        @test all(isapprox.(mean.(obs_distributions(hsmm_est)), mean.(dists); atol=0.1))
        @test all(isapprox.(std.(obs_distributions(hsmm_est)), std.(dists); atol=0.05))
        @test all(
            isapprox.(mean.(duration_distributions(hsmm_est)), mean.(dur_dists); rtol=0.08)
        )
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
        @test all(
            isapprox.(mean.(duration_distributions(hsmm_est)), mean.(dur_dists); rtol=0.05)
        )
        @test all(isapprox.(mean.(obs_distributions(hsmm_est)), mean.(dists); atol=0.05))
    end

    @testset "Exact equivalence with an HMM under geometric sojourns" begin
        hmm = HMM(
            [0.5, 0.3, 0.2],
            [0.7 0.2 0.1; 0.3 0.5 0.2; 0.25 0.25 0.5],
            [Normal(0.0), Normal(5.0), Normal(10.0)],
        )
        function hsmm_from_hmm(hmm)
            A = transition_matrix(hmm)
            return HSMM(
                initialization(hmm),
                [i == j ? 0.0 : A[i, j] / (1 - A[i, i]) for i in 1:3, j in 1:3],
                obs_distributions(hmm),
                [Geometric(1 - A[i, i]) for i in 1:3],
            )
        end
        # Run a fixed number of iterations, so that both runs stop at the same one.
        kwargs = (; max_iterations=10, atol=-Inf, loglikelihood_increasing=false)
        # Short sequences make the censored sojourns matter.
        for (seed, lengths) in ((533, [80, 120]), (534, fill(6, 30)))
            obs_seq, seq_ends = rand_sequences(StableRNG(seed), hmm, lengths)
            hmm_est, logL_hmm = baum_welch(hmm, obs_seq; seq_ends, kwargs...)
            hsmm_est, logL_hsmm = baum_welch(
                hsmm_from_hmm(hmm), obs_seq; seq_ends, kwargs...
            )
            hsmm_ref = hsmm_from_hmm(hmm_est)
            @test logL_hsmm ≈ logL_hmm
            @test initialization(hsmm_est) ≈ initialization(hsmm_ref)
            @test transition_matrix(hsmm_est) ≈ transition_matrix(hsmm_ref)
            @test succprob.(duration_distributions(hsmm_est)) ≈
                succprob.(duration_distributions(hsmm_ref))
            @test mean.(obs_distributions(hsmm_est)) ≈ mean.(obs_distributions(hsmm_ref))
            @test std.(obs_distributions(hsmm_est)) ≈ std.(obs_distributions(hsmm_ref))
        end
        @test_throws ArgumentError baum_welch(hmm, randn(10); max_duration=5)
    end

    @testset "Multiple sequences of differing lengths" begin
        # Vectors of sequence ends go through the threaded loops, tuples do not.
        hsmm, obs_seq, seq_ends = two_state_problem()
        hsmm_vec, logL_vec = baum_welch(hsmm, obs_seq; seq_ends, max_iterations=10)
        hsmm_tup, logL_tup = baum_welch(
            hsmm, obs_seq; seq_ends=Tuple(seq_ends), max_iterations=10
        )
        @test logL_vec ≈ logL_tup
        @test transition_matrix(hsmm_vec) ≈ transition_matrix(hsmm_tup)
        @test succprob.(duration_distributions(hsmm_vec)) ≈
            succprob.(duration_distributions(hsmm_tup))
    end

    @testset "Sparse transition matrix" begin
        trans = sparse([0.0 0.5 0.5; 1.0 0.0 0.0; 1.0 0.0 0.0])
        dists = [Normal(0.0), Normal(3.0), Normal(6.0)]
        durs = fill(Geometric(0.3), 3)
        obs_seq, seq_ends = rand_sequences(
            StableRNG(509), HSMM([0.5, 0.3, 0.2], trans, dists, durs), fill(100, 3)
        )
        init_guess = [0.5, 0.3, 0.2]
        trans_guess = sparse([0.0 0.3 0.7; 1.0 0.0 0.0; 1.0 0.0 0.0])
        dists_guess = [Normal(0.5), Normal(2.5), Normal(5.0)]
        durs_guess = fill(Geometric(0.5), 3)
        hsmm_est, logL_evolution = baum_welch(
            HSMM(init_guess, trans_guess, dists_guess, durs_guess), obs_seq; seq_ends
        )
        hsmm_dense, logL_dense = baum_welch(
            HSMM(init_guess, Matrix(trans_guess), dists_guess, durs_guess),
            obs_seq;
            seq_ends,
        )
        @test logL_evolution ≈ logL_dense
        @test transition_matrix(hsmm_est) isa SparseMatrixCSC
        @test nnz(transition_matrix(hsmm_est)) == nnz(trans)
        @test nnz(log_transition_matrix(hsmm_est)) == nnz(trans)
        @test transition_matrix(hsmm_est) ≈ transition_matrix(hsmm_dense)
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
            trans_est = transition_matrix(hsmm_est)
            dur_dists_est = duration_distributions(hsmm_est)
            @test all(>=(0), diff(logL_evolution))
            @test initialization(hsmm_est) == [1.0, 0.0, 0.0]
            @test trans_est[1:2, :] == [0.0 1.0 0.0; 0.0 0.0 1.0]
            @test trans_est[3, :] == trans_guess[3, :]
            @test dur_dists_est[1:2] != dur_dists[1:2]
            @test dur_dists_est[3] == dur_dists[3]
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
        @test initialization(hsmm_est)[3] == 0
        @test transition_matrix(hsmm_est)[3, :] == trans[3, :]
        @test obs_distributions(hsmm_est)[3] == dists[3]
        @test duration_distributions(hsmm_est)[3] == durs[3]
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
        @test transition_matrix(hsmm_est) isa Matrix{Float32}
        @test duration_distributions(hsmm_est) isa Vector{Geometric{Float64}}
    end

    @testset "Duration counts are required" begin
        hsmm = rand_hsmm(StableRNG(524), 3)
        obs_seq = rand(StableRNG(525), hsmm, 20).obs_seq
        seq_ends = (20,)
        storage = initialize_forward_backward(hsmm, obs_seq, Fill(nothing, 20); seq_ends)
        @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
        # With room for them, `forward_backward!` fills them by default, like `baum_welch`.
        storage = initialize_forward_backward(
            hsmm, obs_seq, Fill(nothing, 20); seq_ends, duration_counts=true
        )
        forward_backward!(storage, hsmm, obs_seq, Fill(nothing, 20); seq_ends)
        hsmm_est = deepcopy(hsmm)
        fit!(hsmm_est, storage, obs_seq; seq_ends)
        hsmm_bw, _ = baum_welch(hsmm, obs_seq; max_iterations=1)
        @test succprob.(duration_distributions(hsmm_est)) ≈
            succprob.(duration_distributions(hsmm_bw))
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
        @test_throws ArgumentError forward_backward!(
            storage, hsmm, obs_seq, control_seq; seq_ends, transition_marginals=true
        )
        # Without room for them, `forward_backward!` skips them by default.
        forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
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
        forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
        @test_throws ArgumentError fit!(hsmm, storage, obs_seq; seq_ends)
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
