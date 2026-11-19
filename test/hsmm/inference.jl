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

    @testset "Empty sequences are rejected" begin
        hsmm = rand_hsmm(StableRNG(5), 2)
        obs_seq, state_seq = randn(StableRNG(6), 5), [1, 1, 2, 2, 1]
        for seq_ends in ([0, 5], [2, 2, 5], (5, 5))
            @test_throws ArgumentError forward(hsmm, obs_seq; seq_ends)
            @test_throws ArgumentError viterbi(hsmm, obs_seq; seq_ends)
            @test_throws ArgumentError forward_backward(hsmm, obs_seq; seq_ends)
            @test_throws ArgumentError logdensityof(hsmm, obs_seq; seq_ends)
            @test_throws ArgumentError joint_logdensityof(
                hsmm, obs_seq, state_seq; seq_ends
            )
            @test_throws ArgumentError baum_welch(hsmm, obs_seq; seq_ends)
        end
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
        # `-Inf` with NaN partials, which must not leak into the gradient.
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

    @testset "Zygote gradients" begin
        function build(θ)
            init = [θ[1], θ[2], 1 - θ[1] - θ[2]]
            trans = [0 θ[3] 1-θ[3]; θ[4] 0 1-θ[4]; θ[5] 1-θ[5] 0]
            dists = [Normal(θ[6]), Normal(θ[7]), Normal(θ[8])]
            return HSMM(init, trans, dists, Geometric.(θ[9:11]))
        end
        θ = [0.5, 0.3, 0.6, 0.4, 0.7, -2.0, 0.0, 2.0, 0.3, 0.2, 0.5]
        obs_seq = rand(StableRNG(1), build(θ), 60).obs_seq
        for kwargs in (
            (;),
            (; seq_ends=[20, 21, 45, 60]),
            (; seq_ends=[20, 21, 45, 60], max_duration=5),
        )
            f(θ) = logdensityof(build(θ), obs_seq; kwargs...)
            @test only(Zygote.gradient(f, θ)) ≈ ForwardDiff.gradient(f, θ)
        end

        # At a zero initial probability, the derivative is one-sided.
        θ0 = [θ[1]; 0.0; θ[3:end]]
        f0(θ) = logdensityof(build(θ), obs_seq)
        h = 1e-7
        one_sided = (f0(θ0 + h * (1:11 .== 2)) - f0(θ0)) / h
        @test only(Zygote.gradient(f0, θ0))[2] ≈ one_sided rtol = 1e-5

        # The duration counts are summed over all start times.
        @test_throws ArgumentError Zygote.gradient(
            θ -> logdensityof(build(θ), obs_seq, collect(1:60)), θ
        )
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
        # Distinct observations avoid exact ties between optimal paths.
        obs_seq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 3.0, 3.0]
        T = length(obs_seq)
        results = [viterbi(hsmm, obs_seq; max_duration) for max_duration in 1:T]
        logLs = [only(logL) for (_, logL) in results]
        @test issorted(logLs)
        for D in 2:T
            # The optimum improves exactly when it needs a sojourn of length `D`.
            q = first(results[D])
            @test (logLs[D] > logLs[D - 1]) == (longest_segment(q) == D)
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
