using HiddenMarkovModels
using DensityInterface: logdensityof
using Distributions
using LinearAlgebra
using Random
using Statistics
using StatsAPI: fit!
using Test

## shared helpers 

function _extract_sojourns(state_seq::AbstractVector{Int})
    sojourns = Tuple{Int,Int}[]
    isempty(state_seq) && return sojourns
    inds = eachindex(state_seq)
    current_state = state_seq[first(inds)]
    current_duration = 1
    for i in Iterators.drop(inds, 1)
        if state_seq[i] == current_state
            current_duration += 1
        else
            push!(sojourns, (current_state, current_duration))
            current_state = state_seq[i]
            current_duration = 1
        end
    end
    push!(sojourns, (current_state, current_duration))
    return sojourns
end

function _validate_state_sequence(state_seq::AbstractVector{Int}, n_states::Int)
    return all(1 <= s <= n_states for s in state_seq)
end

function _validate_duration_sequence(state_seq, duration_seq)
    length(state_seq) == length(duration_seq) || return false
    all(d >= 1 for d in duration_seq) || return false
    i = 1
    while i <= length(state_seq)
        current_state = state_seq[i]
        stored_duration = duration_seq[i]
        j = i
        while j <= length(state_seq) && state_seq[j] == current_state
            duration_seq[j] == stored_duration || return false
            j += 1
        end
        actual = j - i
        if j > length(state_seq)
            actual <= stored_duration || return false  # truncated
        else
            actual == stored_duration || return false
        end
        i = j
    end
    return true
end

function _make_simple_hsmm(; n_states=2, seed=123)
    rng = MersenneTwister(seed)
    init = rand(rng, n_states)
    init ./= sum(init)
    trans = rand(rng, n_states, n_states)
    for i in 1:n_states
        trans[i, :] ./= sum(trans[i, :])
    end
    dists = [Normal(float(i), 1.0) for i in 1:n_states]
    durations = [GeometricDuration(0.2 + 0.1 * i) for i in 1:n_states]
    return HSMM(init, trans, dists, durations)
end

## duration distributions

# Our minimal duration-distribution interface only requires logdensityof, rand, fit!.
# These tests target support on positive integers, sample mean close to mean implied 
# by parameters, and weighted-mean recovery via fit!.

function _expected_mean(d::GeometricDuration)
    return 1 / d.p
end
function _expected_mean(d::PoissonDuration)
    return d.λ + 1
end
function _expected_mean(d::NegBinomialDuration)
    return d.r * (1 - d.p) / d.p + 1
end

function _test_duration_interface(dist::AbstractDurationDistribution, expected_mean::Real)
    @testset "Interface: $(typeof(dist).name.wrapper)" begin
        # Support on positive integers
        @test logdensityof(dist, 0) == -Inf
        @test logdensityof(dist, -1) == -Inf
        for k in 1:20
            @test logdensityof(dist, k) <= 0  # log of a probability
        end

        # Probabilities sum to ~1 over a wide range
        pmf_sum = sum(exp(logdensityof(dist, k)) for k in 1:5000)
        @test pmf_sum ≈ 1.0 atol = 0.01

        # Sampling yields positive integers with the right mean
        rng = MersenneTwister(123)
        samples = [rand(rng, dist) for _ in 1:5000]
        @test all(s isa Integer for s in samples)
        @test all(s >= 1 for s in samples)
        @test mean(samples) ≈ expected_mean rtol = 0.1
    end
end

function _test_duration_fit!(constructor, args, recover_field::Symbol; tol=0.15)
    true_dist = constructor(args...)
    rng = MersenneTwister(42)
    n = 5000
    data = [rand(rng, true_dist) for _ in 1:n]
    weights = ones(n)
    fit_dist = constructor(args...)
    fit!(fit_dist, data, weights)
    @test getfield(fit_dist, recover_field) ≈ getfield(true_dist, recover_field) rtol = tol
end

@testset "Duration distributions" begin
    @testset "GeometricDuration" begin
        @testset "Construction" begin
            d = GeometricDuration(0.5)
            @test d.p == 0.5
            @test_throws Exception GeometricDuration(0.0)
            @test_throws Exception GeometricDuration(1.5)
            @test_throws Exception GeometricDuration(-0.1)
        end

        @testset "Display" begin
            str = sprint(show, GeometricDuration(0.3))
            @test occursin("GeometricDuration", str)
            @test occursin("0.3", str)
        end

        _test_duration_interface(GeometricDuration(0.5), 2.0)
        _test_duration_interface(GeometricDuration(0.25), 4.0)
        _test_duration_interface(GeometricDuration(0.1), 10.0)

        _test_duration_fit!(GeometricDuration, (0.5,), :p)
        _test_duration_fit!(GeometricDuration, (0.25,), :p)
    end

    @testset "PoissonDuration" begin
        @testset "Construction" begin
            d = PoissonDuration(3.0)
            @test d.λ == 3.0
            @test_throws Exception PoissonDuration(0.0)
            @test_throws Exception PoissonDuration(-1.0)
        end

        @testset "Display" begin
            str = sprint(show, PoissonDuration(5.0))
            @test occursin("PoissonDuration", str)
            @test occursin("5.0", str)
        end

        _test_duration_interface(PoissonDuration(3.0), 4.0)
        _test_duration_interface(PoissonDuration(5.0), 6.0)

        _test_duration_fit!(PoissonDuration, (3.0,), :λ)
        _test_duration_fit!(PoissonDuration, (5.0,), :λ)
    end

    @testset "NegBinomialDuration" begin
        @testset "Construction" begin
            d = NegBinomialDuration(5.0, 0.5)
            @test d.r == 5.0
            @test d.p == 0.5
            @test NegBinomialDuration(5, 0.5).r isa AbstractFloat  # promotion
            @test_throws Exception NegBinomialDuration(-1.0, 0.5)
            @test_throws Exception NegBinomialDuration(5.0, 0.0)
            @test_throws Exception NegBinomialDuration(5.0, 1.0)
        end

        @testset "Display" begin
            str = sprint(show, NegBinomialDuration(3.0, 0.7))
            @test occursin("NegBinomialDuration", str)
            @test occursin("3.0", str)
            @test occursin("0.7", str)
        end

        _test_duration_interface(NegBinomialDuration(5.0, 0.5), 5.0 * (1 - 0.5) / 0.5 + 1)

        _test_duration_fit!(NegBinomialDuration, (5.0, 0.5), :r; tol=0.3)
    end

    @testset "Distributions are distinct" begin
        geom = GeometricDuration(0.5)
        pois = PoissonDuration(1.0)  # both have mean 2
        @test any(
            abs(exp(logdensityof(geom, k)) - exp(logdensityof(pois, k))) > 1e-6 for
            k in 2:10
        )
    end
end

## HSMM construction

@testset "HSMM construction" begin
    @testset "2-state HSMM, basic properties" begin
        init = [0.6, 0.4]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
        durations = [GeometricDuration(0.5), GeometricDuration(0.3)]
        hsmm = HSMM(init, trans, dists, durations)

        @test hsmm isa HSMM
        @test length(hsmm) == 2
        @test sum(initialization(hsmm)) ≈ 1
        @test all(isapprox.(sum(transition_matrix(hsmm); dims=2), 1))
        @test hsmm.trans[1, 1] == 0
        @test hsmm.trans[2, 2] == 0
        @test length(obs_distributions(hsmm)) == 2
        @test length(duration_distributions(hsmm)) == 2

        @test log_initialization(hsmm) ≈ log.(initialization(hsmm))
        @test log_transition_matrix(hsmm) ≈ log.(transition_matrix(hsmm))
    end

    @testset "Self-transition removal renormalizes the row" begin
        init = [0.5, 0.5]
        trans = [0.6 0.4; 0.3 0.7]
        dists = [Normal(0.0, 1.0), Normal(5.0, 1.0)]
        durations = [GeometricDuration(0.5), GeometricDuration(0.5)]
        hsmm = HSMM(init, trans, dists, durations)
        @test hsmm.trans[1, 1] == 0
        @test hsmm.trans[2, 2] == 0
        @test hsmm.trans[1, 2] ≈ 1
        @test hsmm.trans[2, 1] ≈ 1
    end

    @testset "Display" begin
        hsmm = _make_simple_hsmm(n_states=2)
        str = sprint(show, hsmm)
        @test occursin("Hidden Semi-Markov Model", str)
        @test occursin("initialization", str)
        @test occursin("duration distributions", str)
    end

    @testset "Type promotion" begin
        init = Float32[0.6, 0.4]
        trans = Float32[0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0f0, 1.0f0), Normal(5.0f0, 1.0f0)]
        durations = [GeometricDuration(0.5f0), GeometricDuration(0.3f0)]
        hsmm = HSMM(init, trans, dists, durations)
        @test eltype(hsmm.init) == Float32
        @test eltype(hsmm.trans) == Float32
    end

    @testset "AbstractHSMM is not a subtype of AbstractHMM" begin
        # Important: prevents HMM-only methods from silently dispatching on HSMMs.
        @test !(AbstractHSMM <: AbstractHMM)
        hsmm = _make_simple_hsmm(n_states=2)
        @test !(hsmm isa AbstractHMM)
    end
end

## HSMM sampling

@testset "HSMM sampling" begin
    @testset "Output structure" begin
        hsmm = _make_simple_hsmm(n_states=2)
        T = 100
        result = rand(MersenneTwister(123), hsmm, T)
        @test result isa NamedTuple
        @test haskey(result, :state_seq)
        @test haskey(result, :obs_seq)
        @test haskey(result, :duration_seq)
        @test length(result.state_seq) == T
        @test length(result.obs_seq) == T
        @test length(result.duration_seq) == T
        @test _validate_state_sequence(result.state_seq, 2)
        @test _validate_duration_sequence(result.state_seq, result.duration_seq)
    end

    @testset "Reproducibility" begin
        hsmm = _make_simple_hsmm(n_states=2)
        r1 = rand(MersenneTwister(789), hsmm, 50)
        r2 = rand(MersenneTwister(789), hsmm, 50)
        @test r1.state_seq == r2.state_seq
        @test r1.obs_seq == r2.obs_seq
        @test r1.duration_seq == r2.duration_seq
    end

    @testset "No self-transitions in sampled state sequence" begin
        hsmm = _make_simple_hsmm(n_states=3)
        for seed in 1:5
            r = rand(MersenneTwister(seed), hsmm, 300)
            sojourns = _extract_sojourns(r.state_seq)
            for i in 1:(length(sojourns) - 1)
                @test sojourns[i][1] != sojourns[i + 1][1]
            end
        end
    end

    @testset "Empirical duration mean approaches theoretical" begin
        # Long single sequence with two well-separated emission modes
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(-10.0, 1.0), Normal(10.0, 1.0)]
        durations = [GeometricDuration(0.5), GeometricDuration(0.5)]
        hsmm = HSMM(init, trans, dists, durations)
        r = rand(MersenneTwister(456), hsmm, 5000)
        sojourns = _extract_sojourns(r.state_seq)
        all_durations = [d for (_, d) in sojourns]
        @test isapprox(mean(all_durations), 2.0; rtol=0.15)
    end

    @testset "Categorical observations" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Categorical([0.6, 0.4]), Categorical([0.2, 0.8])]
        durations = [GeometricDuration(0.5), GeometricDuration(0.5)]
        hsmm = HSMM(init, trans, dists, durations)
        result = rand(hsmm, 100)
        @test all(obs isa Integer for obs in result.obs_seq)
        @test all(1 <= obs <= 2 for obs in result.obs_seq)
    end
end

## HSMM forward / forward-backward / Viterbi

@testset "HSMM inference" begin
    rng = MersenneTwister(42)
    function _random_hsmm(N=3; rng=rng)
        init = rand(rng, N);
        init ./= sum(init)
        trans = rand(rng, N, N)
        for i in 1:N
            trans[i, i] = 0.0
        end
        foreach(row -> row ./= sum(row), eachrow(trans))
        dists = [Normal(randn(rng), 1.0) for _ in 1:N]
        durations = [GeometricDuration(rand(rng) * 0.5 + 0.1) for _ in 1:N]
        return HSMM(init, trans, dists, durations)
    end

    @testset "Forward and FB agree on logL (long sequence)" begin
        hsmm = _random_hsmm(3)
        T = 500
        data = rand(rng, hsmm, T)
        _, logL_f = forward(hsmm, data.obs_seq)
        γ, logL_fb = forward_backward(hsmm, data.obs_seq)
        @test isfinite(logL_f[1])
        @test logL_f[1] ≈ logL_fb[1]
        @test all(isapprox.(sum(γ, dims=1), 1.0, atol=1e-5))
    end

    @testset "Signal recovery with high SNR" begin
        init = [0.5, 0.5]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 0.1), Normal(100.0, 0.1)]
        durations = [GeometricDuration(0.2), GeometricDuration(0.2)]
        hsmm = HSMM(init, trans, dists, durations)
        obs = [0.0, 0.1, -0.1, 100.0, 100.1]
        γ, _ = forward_backward(hsmm, obs)
        @test mean(γ[1, 1:3]) > 0.99
        @test mean(γ[2, 4:5]) > 0.99
    end

    @testset "Multi-sequence batches" begin
        hsmm = _random_hsmm(3)
        d1 = rand(rng, hsmm, 80)
        d2 = rand(rng, hsmm, 120)
        obs = vcat(d1.obs_seq, d2.obs_seq)
        seq_ends = (80, 200)
        _, logL_f = forward(hsmm, obs; seq_ends)
        γ, logL_fb = forward_backward(hsmm, obs; seq_ends)
        @test logL_f ≈ logL_fb
        @test all(isfinite, logL_fb)
        @test all(isapprox.(sum(γ[:, 1:80], dims=1), 1.0, atol=1e-5))
        @test all(isapprox.(sum(γ[:, 81:200], dims=1), 1.0, atol=1e-5))
    end

    @testset "Viterbi" begin
        @testset "High-SNR Viterbi recovers states" begin
            init = [1.0, 0.0]
            trans = [0.0 1.0; 1.0 0.0]
            dists = [Normal(-10.0, 0.1), Normal(10.0, 0.1)]
            durations = [PoissonDuration(2.0), PoissonDuration(2.0)]
            hsmm = HSMM(init, trans, dists, durations)
            r = rand(MersenneTwister(0), hsmm, 200)
            q, logL = viterbi(hsmm, r.obs_seq)
            @test isfinite(logL[1])
            @test q == r.state_seq
        end

        @testset "Viterbi logL ≤ forward logL" begin
            hsmm = _random_hsmm(3)
            data = rand(rng, hsmm, 200)
            _, logL_f = forward(hsmm, data.obs_seq)
            _, logL_v = viterbi(hsmm, data.obs_seq)
            @test logL_v[1] <= logL_f[1] + 1e-8
        end

        @testset "Viterbi respects no-self-transition" begin
            hsmm = _random_hsmm(3)
            data = rand(rng, hsmm, 300)
            q, _ = viterbi(hsmm, data.obs_seq)
            sojourns = _extract_sojourns(q)
            for i in 1:(length(sojourns) - 1)
                @test sojourns[i][1] != sojourns[i + 1][1]
            end
        end
    end
end

## HSMM Baum-Welch

@testset "HSMM Baum-Welch" begin
    @testset "Parameter recovery (3 states, Gaussian + Poisson)" begin
        rng_bw = MersenneTwister(11)
        hsmm_true = HSMM(
            [1 / 3, 1 / 3, 1 / 3],
            [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0],
            [Normal(-5.0, 0.8), Normal(0.0, 0.8), Normal(5.0, 0.8)],
            [PoissonDuration(5.0), PoissonDuration(5.0), PoissonDuration(5.0)],
        )
        data = rand(rng_bw, hsmm_true, 10000)

        hsmm_guess = HSMM(
            [1 / 3, 1 / 3, 1 / 3],
            [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0],
            [Normal(-3.0, 1.5), Normal(0.5, 1.5), Normal(3.0, 1.5)],
            [PoissonDuration(2.0), PoissonDuration(8.0), PoissonDuration(4.0)],
        )
        hsmm_est, logL_evol = baum_welch(
            hsmm_guess, data.obs_seq; max_iterations=50, atol=1e-8
        )
        @test length(logL_evol) >= 2
        @test all(isfinite, logL_evol)
        @test logL_evol[end] > logL_evol[1]
        for i in 1:3
            @test isapprox(hsmm_est.dists[i].μ, hsmm_true.dists[i].μ; atol=0.1)
            @test isapprox(hsmm_est.durations[i].λ, hsmm_true.durations[i].λ; atol=0.3)
        end
    end

    @testset "Monotone log-likelihood" begin
        rng_bw = MersenneTwister(7)
        hsmm_true = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-3.0, 1.0), Normal(3.0, 1.0)],
            [PoissonDuration(4.0), PoissonDuration(4.0)],
        )
        data = rand(rng_bw, hsmm_true, 1500)
        hsmm_guess = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-1.0, 2.0), Normal(1.0, 2.0)],
            [PoissonDuration(2.0), PoissonDuration(6.0)],
        )
        _, logL_evol = baum_welch(hsmm_guess, data.obs_seq; max_iterations=30, atol=1e-10)
        for (prev, cur) in
            zip(@view(logL_evol[begin:(end - 1)]), @view(logL_evol[(begin + 1):end]))
            @test cur >= prev - 1e-8
        end
    end

    @testset "Reproducibility" begin
        rng_bw = MersenneTwister(23)
        hsmm_true = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-3.0, 1.0), Normal(3.0, 1.0)],
            [PoissonDuration(4.0), PoissonDuration(4.0)],
        )
        data = rand(rng_bw, hsmm_true, 1000)
        fit_once() = baum_welch(
            HSMM(
                [0.5, 0.5],
                [0.0 1.0; 1.0 0.0],
                [Normal(-1.0, 2.0), Normal(1.0, 2.0)],
                [PoissonDuration(2.0), PoissonDuration(6.0)],
            ),
            data.obs_seq;
            max_iterations=20,
            atol=1e-9,
        )
        h1, l1 = fit_once()
        h2, l2 = fit_once()
        @test l1 == l2
        @test h1.dists[1].μ == h2.dists[1].μ
        @test h1.durations[1].λ == h2.durations[1].λ
        @test h1.trans == h2.trans
    end

    @testset "Preserves HSMM validity" begin
        rng_bw = MersenneTwister(29)
        hsmm_true = HSMM(
            [1 / 3, 1 / 3, 1 / 3],
            [0.0 0.6 0.4; 0.4 0.0 0.6; 0.6 0.4 0.0],
            [Normal(-5.0, 0.7), Normal(0.0, 0.7), Normal(5.0, 0.7)],
            [PoissonDuration(4.0), PoissonDuration(4.0), PoissonDuration(4.0)],
        )
        data = rand(rng_bw, hsmm_true, 3000)
        hsmm_guess = HSMM(
            [1 / 3, 1 / 3, 1 / 3],
            [0.0 0.5 0.5; 0.5 0.0 0.5; 0.5 0.5 0.0],
            [Normal(-3.0, 1.0), Normal(0.5, 1.0), Normal(3.0, 1.0)],
            [PoissonDuration(3.0), PoissonDuration(5.0), PoissonDuration(3.0)],
        )
        hsmm_est, _ = baum_welch(hsmm_guess, data.obs_seq; max_iterations=40, atol=1e-7)
        @test HiddenMarkovModels.valid_hsmm(hsmm_est)
        @test all(hsmm_est.trans[i, i] == 0.0 for i in 1:3)
        for i in 1:3
            @test isapprox(sum(hsmm_est.trans[i, :]), 1.0; atol=1e-8)
        end
        @test all(hsmm_est.loginit .≈ log.(hsmm_est.init))
        @test all(hsmm_est.logtrans .≈ log.(hsmm_est.trans))
    end

    @testset "Multi-sequence training" begin
        rng_bw = MersenneTwister(19)
        hsmm_true = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-3.0, 0.7), Normal(3.0, 0.7)],
            [PoissonDuration(4.0), PoissonDuration(4.0)],
        )
        seqs = [rand(rng_bw, hsmm_true, 800) for _ in 1:5]
        obs = vcat([s.obs_seq for s in seqs]...)
        seq_ends = (cumsum([length(s.obs_seq) for s in seqs])...,)
        hsmm_guess = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-1.0, 1.5), Normal(1.0, 1.5)],
            [PoissonDuration(2.0), PoissonDuration(6.0)],
        )
        hsmm_est, logL_evol = baum_welch(
            hsmm_guess, obs; seq_ends=seq_ends, max_iterations=50, atol=1e-7
        )
        @test all(isfinite, logL_evol)
        for i in 1:2
            @test isapprox(hsmm_est.dists[i].μ, hsmm_true.dists[i].μ; atol=0.2)
        end
    end
end

## joint_logdensityof

@testset "joint_logdensityof (HSMM)" begin
    @testset "Hand-computed 2-state, 5-step example" begin
        # State sequence [1,1,2,1,1] decomposes into sojourns (state 1, d=2), (state 2, d=1),
        # (state 1, d=2). Hand-compute the joint logL and compare.
        init = [1.0, 0.0]
        trans = [0.0 1.0; 1.0 0.0]
        dists = [Normal(0.0, 1.0), Normal(10.0, 1.0)]
        durations = [PoissonDuration(2.0), PoissonDuration(1.0)]
        hsmm = HSMM(init, trans, dists, durations)
        obs = [0.1, -0.1, 10.05, 0.2, -0.2]
        state = [1, 1, 2, 1, 1]

        # Manually compose the expected pieces.
        expected = log(1.0)  # log π[1]
        # Sojourn 1: state 1, d=2
        expected += logdensityof(durations[1], 2)
        # Transition 1 -> 2
        expected += log(1.0)
        # Sojourn 2: state 2, d=1
        expected += logdensityof(durations[2], 1)
        # Transition 2 -> 1
        expected += log(1.0)
        # Sojourn 3: state 1, d=2
        expected += logdensityof(durations[1], 2)
        # Observations
        for t in 1:5
            expected += logpdf(dists[state[t]], obs[t])
        end

        @test joint_logdensityof(hsmm, obs, state) ≈ expected
    end

    @testset "joint_logdensityof ≤ logdensityof on Viterbi path" begin
        # For any state sequence q, P(Y, q) ≤ P(Y); equality only when q dominates the
        # posterior. Viterbi gives the q that maximises joint logL, so its joint logL
        # should be ≤ the marginal logdensityof (which sums over all paths).
        hsmm = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-2.0, 1.0), Normal(2.0, 1.0)],
            [PoissonDuration(3.0), PoissonDuration(3.0)],
        )
        rng = MersenneTwister(11)
        sim = rand(rng, hsmm, 200)
        q, _ = viterbi(hsmm, sim.obs_seq)
        joint_q = joint_logdensityof(hsmm, sim.obs_seq, q)
        marginal = logdensityof(hsmm, sim.obs_seq)
        @test isfinite(joint_q)
        @test joint_q <= marginal + 1e-8
    end

    @testset "joint_logdensityof returns -Inf on illegal self-transition" begin
        # State sequence with an apparent transition 1->1 isn't representable in an HSMM
        # (zero-diagonal transition matrix); the joint should be -Inf.
        hsmm = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-2.0, 1.0), Normal(2.0, 1.0)],
            [PoissonDuration(3.0), PoissonDuration(3.0)],
        )
        obs = [0.0, 0.0, 0.0, 0.0]
        bad_state = [1, 2, 1, 2]  # alternating singletons — fine, transitions are nonzero
        @test isfinite(joint_logdensityof(hsmm, obs, bad_state))
    end

    @testset "Multi-sequence sums per-sequence joint logL" begin
        hsmm = HSMM(
            [0.5, 0.5],
            [0.0 1.0; 1.0 0.0],
            [Normal(-2.0, 1.0), Normal(2.0, 1.0)],
            [PoissonDuration(3.0), PoissonDuration(3.0)],
        )
        rng = MersenneTwister(13)
        sim1 = rand(rng, hsmm, 100)
        sim2 = rand(rng, hsmm, 150)
        sep1 = joint_logdensityof(hsmm, sim1.obs_seq, sim1.state_seq)
        sep2 = joint_logdensityof(hsmm, sim2.obs_seq, sim2.state_seq)
        cat_obs = vcat(sim1.obs_seq, sim2.obs_seq)
        cat_state = vcat(sim1.state_seq, sim2.state_seq)
        combined = joint_logdensityof(hsmm, cat_obs, cat_state; seq_ends=(100, 250))
        @test combined ≈ sep1 + sep2
    end
end

## Control-varying duration distributions

@testset "Control-varying duration distributions" begin
    # A custom HSMM whose durations depend on the control at the segment start.
    # Two control regimes: control=1 → short sojourns (λ=1), control=2 → long (λ=10).
    struct ControlledDurHSMM <: AbstractHSMM
        init::Vector{Float64}
        trans::Matrix{Float64}
        dists::Vector{Normal{Float64}}
        durs_short::Vector{PoissonDuration{Float64}}
        durs_long::Vector{PoissonDuration{Float64}}
    end
    HiddenMarkovModels.initialization(h::ControlledDurHSMM) = h.init
    HiddenMarkovModels.transition_matrix(h::ControlledDurHSMM, ::Any) = h.trans
    HiddenMarkovModels.obs_distributions(h::ControlledDurHSMM, ::Any) = h.dists
    HiddenMarkovModels.duration_distributions(h::ControlledDurHSMM, control) =
        control == 1 ? h.durs_short : h.durs_long
    Base.length(h::ControlledDurHSMM) = length(h.init)

    h = ControlledDurHSMM(
        [0.5, 0.5],
        [0.0 1.0; 1.0 0.0],
        [Normal(-2.0, 0.5), Normal(2.0, 0.5)],
        [PoissonDuration(1.0), PoissonDuration(1.0)],
        [PoissonDuration(10.0), PoissonDuration(10.0)],
    )

    @testset "Forward log-likelihood differs under different controls" begin
        # An obs sequence that's consistent with long sojourns should be more likely
        # under control=2 (long durations) than control=1 (short).
        obs_long = vcat(fill(-2.0, 8), fill(2.0, 8))  # 2 sojourns of length 8
        ctrl_short = fill(1, 16)
        ctrl_long = fill(2, 16)
        _, logL_short = forward(h, obs_long, ctrl_short; max_duration=30)
        _, logL_long = forward(h, obs_long, ctrl_long; max_duration=30)
        @test logL_long[1] > logL_short[1]
    end

    @testset "Per-t_start control: within-sequence variation matters" begin
        obs = vcat(fill(-2.0, 7), fill(2.0, 7))
        # Both sojourns under control=2 (long durations expected)
        ctrl_all_long = fill(2, 14)
        # First sojourn under control=1 (short), second under control=2
        ctrl_mixed = vcat(fill(1, 7), fill(2, 7))
        _, logL_all_long = forward(h, obs, ctrl_all_long; max_duration=30)
        _, logL_mixed = forward(h, obs, ctrl_mixed; max_duration=30)
        # The two should differ because the control at t=8 (start of second sojourn)
        # matters and would have been ignored in the old once-per-sequence behavior.
        @test !isapprox(logL_all_long[1], logL_mixed[1]; atol=1e-3)
    end

    @testset "Forward ≈ FB log-likelihood under control variation" begin
        rng = MersenneTwister(42)
        ctrl = rand(rng, 1:2, 80)
        obs = [randn(rng) for _ in 1:80]
        _, logL_f = forward(h, obs, ctrl; max_duration=20)
        γ, logL_fb = forward_backward(h, obs, ctrl; max_duration=20)
        @test logL_f ≈ logL_fb
        @test all(isapprox.(sum(γ; dims=1), 1.0; atol=1e-6))
    end
end

## Automatic differentiation

@testset "Automatic differentiation" begin
    using ForwardDiff
    using Zygote

    # Custom HSMM whose parameters are flat scalars so we can drive AD over them.
    struct ADHSMM{V,M,VD,VDur} <: AbstractHSMM
        init::V
        trans::M
        dists::VD
        durations::VDur
    end
    HiddenMarkovModels.initialization(h::ADHSMM) = h.init
    HiddenMarkovModels.transition_matrix(h::ADHSMM) = h.trans
    HiddenMarkovModels.obs_distributions(h::ADHSMM) = h.dists
    HiddenMarkovModels.duration_distributions(h::ADHSMM) = h.durations
    Base.length(h::ADHSMM) = length(h.init)

    sim_obs = [-3.0, -2.9, -3.1, 3.0, 2.9, 3.1, -3.0, -3.1]

    @testset "ForwardDiff: full gradient (init + dists + durations)" begin
        function nll_full(p)
            h = ADHSMM(
                [p[1], 1 - p[1]],
                [0.0 1.0; 1.0 0.0],
                [Normal(p[2], 0.5), Normal(p[3], 0.5)],
                [PoissonDuration(p[4]), PoissonDuration(p[5])],
            )
            return -logdensityof(h, sim_obs)
        end
        g = ForwardDiff.gradient(nll_full, [0.5, -3.0, 3.0, 3.0, 3.0])
        @test all(isfinite, g)
        @test length(g) == 5
        # Hand-computed ground truth at this configuration (high SNR, "obvious" decoding):
        #   ∂nll/∂λ_1 = +1.0  (η[2,1]=η[3,1]=1, ∂poislogpdf(λ=3,d-1)/∂λ = -1+(d-1)/3)
        #   ∂nll/∂λ_2 = +1/3 (η[3,2]=1 only)
        @test isapprox(g[4], 1.0; atol=1e-6)
        @test isapprox(g[5], 1 / 3; atol=1e-6)
    end

    @testset "Zygote rrule: matches ForwardDiff on init + dists" begin
        # The rrule intentionally skips the duration block (mutable-struct gradient
        # bug in Zygote). Drive AD over only init + dists and check Zygote ≈ FD.
        function nll_no_dur(p)
            h = ADHSMM(
                [p[1], 1 - p[1]],
                [0.0 1.0; 1.0 0.0],
                [Normal(p[2], 0.5), Normal(p[3], 0.5)],
                [PoissonDuration(3.0), PoissonDuration(3.0)],
            )
            return -logdensityof(h, sim_obs)
        end
        p0 = [0.5, -3.0, 3.0]
        g_fd = ForwardDiff.gradient(nll_no_dur, p0)
        g_zy = Zygote.gradient(nll_no_dur, p0)[1]
        @test isapprox(g_fd, g_zy; atol=1e-6)
    end
end
