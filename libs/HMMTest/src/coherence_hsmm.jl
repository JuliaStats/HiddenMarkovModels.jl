function test_coherent_algorithms(
    rng::AbstractRNG,
    hsmm::AbstractHSMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    hsmm_guess::Union{Nothing,AbstractHSMM}=nothing,
    max_duration::Int=50,
    atol::Real=0.15,
    init::Bool=true,
)
    @testset "Coherence (HSMM)" begin
        simulations = map(eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hsmm, control_seq[t1:t2])
        end

        state_seqs = [sim.state_seq for sim in simulations]
        obs_seqs = [sim.obs_seq for sim in simulations]

        state_seq = reduce(vcat, state_seqs)
        obs_seq = reduce(vcat, obs_seqs)

        logL = logdensityof(hsmm, obs_seq, control_seq; seq_ends, max_duration)
        logL_joint = joint_logdensityof(hsmm, obs_seq, state_seq, control_seq; seq_ends)

        q, logL_viterbi = viterbi(hsmm, obs_seq, control_seq; seq_ends, max_duration)
        @test sum(logL_viterbi) >= logL_joint - 1e-8
        @test sum(logL_viterbi) ≈
            joint_logdensityof(hsmm, obs_seq, q, control_seq; seq_ends)

        log_α, logL_forward = forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
        @test sum(logL_forward) ≈ logL

        γ, logL_forward_backward = forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        @test sum(logL_forward_backward) ≈ logL

        # HSMM's log_α[:, t] is `log P(Y_{1:t}, segment ENDS at t in state i)` — not a
        # filtered marginal like in an HMM. At each sequence boundary the last segment
        # is by construction forced to end there, so renormalizing log_α[:, t2] across
        # states gives the smoothed marginal γ[:, t2].
        for k in eachindex(seq_ends)
            tk = seq_ends[k]
            col = view(log_α, :, tk)
            normalizer = logsumexp_local(col)
            α_t2 = exp.(col .- normalizer)
            @test isapprox(α_t2, γ[:, tk]; atol)
        end

        if !isnothing(hsmm_guess)
            hsmm_est, logL_evolution = baum_welch(
                hsmm_guess, obs_seq, control_seq; seq_ends, max_duration
            )
            @test all(>=(-1e-8), diff(logL_evolution))
            @testset "No flip" begin
                test_equal_hsmms(hsmm, hsmm_est, control_seq[1:3]; atol, init)
            end
            @testset "Flip" begin
                test_equal_hsmms(hsmm, hsmm_guess, control_seq[1:3]; atol, init, flip=true)
            end
        end
    end
end

# Local log-sum-exp helper to avoid a StatsFuns dependency just for this assertion.
function logsumexp_local(v::AbstractVector)
    m = maximum(v)
    isfinite(m) || return m
    s = zero(m)
    @simd for x in v
        s += exp(x - m)
    end
    return m + log(s)
end

function test_equal_hsmms(
    hsmm1::AbstractHSMM,
    hsmm2::AbstractHSMM,
    control_seq::AbstractVector;
    atol::Real,
    init::Bool,
    flip::Bool=false,
)
    @testset "Initialization" begin
        if init
            init1 = initialization(hsmm1)
            init2 = initialization(hsmm2)
            if flip
                @test !isapprox(init1, init2; atol, norm=infnorm)
            else
                @test isapprox(init1, init2; atol, norm=infnorm)
            end
        end
    end

    @testset "Transition matrix" begin
        @testset "Control $control" for control in control_seq
            trans1 = transition_matrix(hsmm1, control)
            trans2 = transition_matrix(hsmm2, control)
            if flip
                @test !isapprox(trans1, trans2; atol, norm=infnorm)
            else
                @test isapprox(trans1, trans2; atol, norm=infnorm)
            end
        end
    end

    @testset "Observation distributions" begin
        @testset "Control $control" for control in control_seq
            dists1 = obs_distributions(hsmm1, control)
            dists2 = obs_distributions(hsmm2, control)
            @testset "State $i" for i in eachindex(dists1, dists2)
                dist1, dist2 = dists1[i], dists2[i]
                @testset "Field $field" for field in fieldnames(typeof(dist1))
                    string(field) in ("μ", "p") || continue
                    x1 = getfield(dist1, field)
                    x2 = getfield(dist2, field)
                    if flip
                        @test !isapprox(x1, x2; atol, norm=infnorm)
                    else
                        @test isapprox(x1, x2; atol, norm=infnorm)
                    end
                end
            end
        end
    end

    @testset "Duration distributions" begin
        @testset "Control $control" for control in control_seq
            durs1 = duration_distributions(hsmm1, control)
            durs2 = duration_distributions(hsmm2, control)
            @testset "State $i" for i in eachindex(durs1, durs2)
                d1, d2 = durs1[i], durs2[i]
                @testset "Field $field" for field in fieldnames(typeof(d1))
                    string(field) in ("λ", "p", "r") || continue
                    x1 = getfield(d1, field)
                    x2 = getfield(d2, field)
                    if flip
                        @test !isapprox(x1, x2; atol, norm=infnorm)
                    else
                        @test isapprox(x1, x2; atol, norm=infnorm)
                    end
                end
            end
        end
    end

    return nothing
end
