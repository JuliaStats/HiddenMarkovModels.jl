infnorm(x) = maximum(abs, x)
infapprox(x, y; atol::Real) = infnorm(x .- y) <= atol

function test_equal_hmms(
    hmm1::AbstractHMM,
    hmm2::AbstractHMM,
    control_seq::AbstractVector;
    atol::Real,
    init::Bool,
    flip::Bool=false,
)
    @testset "Initialization" begin
        if init
            init1 = initialization(hmm1)
            init2 = initialization(hmm2)
            if flip
                @test !infapprox(init1, init2; atol)
            else
                @test infapprox(init1, init2; atol)
            end
        end
    end

    @testset "Transition matrix" begin
        @testset "Control $control" for control in control_seq
            trans1 = transition_matrix(hmm1, control)
            trans2 = transition_matrix(hmm2, control)
            if typeof(trans1) == typeof(trans2)
                @test HMMs.mynnz(trans1) == HMMs.mynnz(trans2)
            end
            if flip
                @test !infapprox(trans1, trans2; atol)
            else
                @test infapprox(trans1, trans2; atol)
            end
        end
    end

    @testset "Observation distributions" begin
        @testset "Control $control" for control in control_seq
            dists1 = obs_distributions(hmm1, control)
            dists2 = obs_distributions(hmm2, control)
            @testset "State $i" for i in eachindex(dists1, dists2)
                dist1, dist2 = dists1[i], dists2[i]
                @testset "Field $field" for field in fieldnames(typeof(dist1))
                    string(field) in ("μ", "p") || continue
                    x1 = getfield(dist1, field)
                    x2 = getfield(dist2, field)
                    if flip
                        @test !infapprox(x1, x2; atol)
                    else
                        @test infapprox(x1, x2; atol)
                    end
                end
            end
        end
    end

    return nothing
end

function test_coherent_algorithms(
    rng::AbstractRNG,
    hmm::AbstractHMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    hmm_guess::Union{Nothing,AbstractHMM}=nothing,
    atol::Real=0.05,
    init::Bool=true,
)
    @testset "Coherence" begin
        simulations = map(eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hmm, control_seq[t1:t2])
        end

        state_seqs = [sim.state_seq for sim in simulations]
        obs_seqs = [sim.obs_seq for sim in simulations]

        state_seq = reduce(vcat, state_seqs)
        obs_seq = reduce(vcat, obs_seqs)

        logL = logdensityof(hmm, obs_seq, control_seq; seq_ends)
        logL_joint = joint_logdensityof(hmm, obs_seq, state_seq, control_seq; seq_ends)

        q, logL_viterbi = viterbi(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_viterbi) > logL_joint
        @test sum(logL_viterbi) ≈ joint_logdensityof(hmm, obs_seq, q, control_seq; seq_ends)

        α, logL_forward = forward(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_forward) .≈ logL

        γ, logL_forward_backward = forward_backward(hmm, obs_seq, control_seq; seq_ends)
        @test sum(logL_forward_backward) ≈ logL
        @test all(α[:, seq_ends[k]] ≈ γ[:, seq_ends[k]] for k in eachindex(seq_ends))

        if !isnothing(hmm_guess)
            hmm_est, logL_evolution = baum_welch(hmm_guess, obs_seq, control_seq; seq_ends)
            @test all(>=(0), diff(logL_evolution))
            @testset "No flip" begin
                test_equal_hmms(hmm, hmm_est, control_seq[1:3]; atol, init)
            end
            @testset "Flip" begin
                test_equal_hmms(hmm, hmm_guess, control_seq[1:3]; atol, init, flip=true)
            end
        end
    end
end
