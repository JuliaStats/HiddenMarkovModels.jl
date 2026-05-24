function test_type_stability(
    rng::AbstractRNG,
    hsmm::AbstractHSMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    hsmm_guess::Union{Nothing,AbstractHSMM}=nothing,
    max_duration::Int=50,
)
    @testset "Type stability (HSMM)" begin
        sim = rand(hsmm, control_seq)
        state_seq, obs_seq = sim.state_seq, sim.obs_seq

        @test_opt target_modules = (HMMs,) rand(hsmm, control_seq)
        @test_call target_modules = (HMMs,) rand(hsmm, control_seq)

        @test_opt target_modules = (HMMs,) logdensityof(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        @test_call target_modules = (HMMs,) logdensityof(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )

        @test_opt target_modules = (HMMs,) joint_logdensityof(
            hsmm, obs_seq, state_seq, control_seq; seq_ends
        )
        @test_call target_modules = (HMMs,) joint_logdensityof(
            hsmm, obs_seq, state_seq, control_seq; seq_ends
        )

        @test_opt target_modules = (HMMs,) forward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        @test_call target_modules = (HMMs,) forward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )

        @test_opt target_modules = (HMMs,) viterbi(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        @test_call target_modules = (HMMs,) viterbi(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )

        @test_opt target_modules = (HMMs,) forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        @test_call target_modules = (HMMs,) forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )

        if !isnothing(hsmm_guess)
            @test_opt target_modules = (HMMs,) baum_welch(
                hsmm, obs_seq, control_seq; seq_ends, max_duration, max_iterations=1
            )
            @test_call target_modules = (HMMs,) baum_welch(
                hsmm, obs_seq, control_seq; seq_ends, max_duration, max_iterations=1
            )
        end
    end
end
