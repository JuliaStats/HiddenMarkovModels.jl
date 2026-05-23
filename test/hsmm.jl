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
# These tests target the contract — sum-to-(close-to-)one over a wide range,
# support on positive integers, sample mean close to mean implied by parameters,
# and weighted-mean recovery via fit!.

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
