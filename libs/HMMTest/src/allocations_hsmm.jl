function test_allocations(
    rng::AbstractRNG,
    hsmm::AbstractHSMM,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    hsmm_guess::Union{Nothing,AbstractHSMM}=nothing,
    max_duration::Int=50,
)
    # Single-sequence tuple disables multithreading so allocs can be measured cleanly.
    seq_ends = ntuple(k -> seq_ends[k], Val(min(1, length(seq_ends))))
    control_seq = control_seq[1:last(seq_ends)]

    @testset "Allocations (HSMM)" begin
        obs_seq = mapreduce(vcat, eachindex(seq_ends)) do k
            t1, t2 = seq_limits(seq_ends, k)
            rand(rng, hsmm, control_seq[t1:t2]).obs_seq
        end

        ## Forward

        f_storage = HMMs.initialize_hsmm_forward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        allocs_f = @ballocated HMMs.forward!(
            $f_storage, $hsmm, $obs_seq, $control_seq; seq_ends=($seq_ends)
        ) evals = 1 samples = 1
        @test allocs_f == 0

        ## Viterbi

        v_storage = HMMs.initialize_hsmm_viterbi(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        allocs_v = @ballocated HMMs.viterbi!(
            $v_storage, $hsmm, $obs_seq, $control_seq; seq_ends=($seq_ends)
        ) evals = 1 samples = 1
        @test allocs_v == 0

        ## Forward-backward

        fb_storage = HMMs.initialize_hsmm_forward_backward(
            hsmm, obs_seq, control_seq; seq_ends, max_duration
        )
        allocs_fb = @ballocated HMMs.forward_backward!(
            $fb_storage, $hsmm, $obs_seq, $control_seq; seq_ends=($seq_ends)
        ) evals = 1 samples = 1
        @test allocs_fb == 0

        ## Baum-Welch (fit! step only — same convention as the HMM allocation test)

        if !isnothing(hsmm_guess)
            fb_storage = HMMs.initialize_hsmm_forward_backward(
                hsmm_guess, obs_seq, control_seq; seq_ends, max_duration
            )
            HMMs.forward_backward!(fb_storage, hsmm, obs_seq, control_seq; seq_ends)
            allocs_bw = @ballocated fit!(
                hsmm_guess_copy, $fb_storage, $obs_seq, $control_seq; seq_ends=($seq_ends)
            ) evals = 1 samples = 1 setup = (hsmm_guess_copy = deepcopy($hsmm_guess))
            @test allocs_bw == 0
        end
    end
end
