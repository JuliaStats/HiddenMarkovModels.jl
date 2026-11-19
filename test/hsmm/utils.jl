using DensityInterface: DensityKind, HasDensity, logdensityof
using Distributions: Binomial, Categorical, Geometric, Normal, Poisson, mean, std, succprob
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels:
    AbstractLatentStateModel,
    HSMMForwardBackwardStorage,
    HSMMForwardStorage,
    HSMMViterbiStorage,
    duration_logdensityof,
    duration_logsurvival,
    duration_logdensity_type,
    elementwise_log,
    Fill,
    StateSegments,
    forward_backward!,
    initialize_forward,
    initialize_forward_backward,
    initialize_viterbi,
    log_initialization,
    log_transition_matrix,
    valid_hsmm,
    viterbi!
using HMMTest: test_allocations, test_type_stability
using LinearAlgebra: Diagonal, diag
using SparseArrays: SparseMatrixCSC, nnz, sparse
using StableRNGs: StableRNG
using Test
using Zygote: Zygote

# Every component depends on the control, so using the wrong control changes the results.
struct ControlledHSMM <: AbstractHSMM
    init::Vector{Float64}
    trans::Vector{Matrix{Float64}}
    dists::Vector{Vector{Normal{Float64}}}
    durs::Vector{Vector{Geometric{Float64}}}
end

HMMs.initialization(hsmm::ControlledHSMM) = hsmm.init
HMMs.transition_matrix(hsmm::ControlledHSMM, control::Integer=1) = hsmm.trans[control]
HMMs.obs_distributions(hsmm::ControlledHSMM, control::Integer=1) = hsmm.dists[control]
function HMMs.duration_distributions(hsmm::ControlledHSMM, control::Integer=1)
    return hsmm.durs[control]
end

function controlled_hsmm()
    init = [0.5, 0.3, 0.2]
    trans = [
        [0.0 0.9 0.1; 0.2 0.0 0.8; 0.6 0.4 0.0], [0.0 0.1 0.9; 0.8 0.0 0.2; 0.3 0.7 0.0]
    ]
    dists = [Normal.([0.0, 2.0, 4.0]), Normal.([4.0, 0.0, 2.0])]
    durs = [Geometric.([0.2, 0.5, 0.8]), Geometric.([0.7, 0.3, 0.4])]
    return ControlledHSMM(init, trans, dists, durs)
end

function rand_hsmm(rng, N)
    init = rand(rng, N)
    trans = [i == j ? 0.0 : rand(rng) for i in 1:N, j in 1:N]
    dists = [Normal(2.0 * i, 1.0) for i in 1:N]
    dur_dists = [Geometric(0.2 + 0.15 * i) for i in 1:N]
    return HSMM(init ./ sum(init), trans ./ sum(trans; dims=2), dists, dur_dists)
end

# The HSMM with geometric sojourns that is equivalent to `hmm`.
function hsmm_from_hmm(hmm)
    A, stay = transition_matrix(hmm), diag(transition_matrix(hmm))
    trans = (A - Diagonal(stay)) ./ (1 .- stay)
    return HSMM(initialization(hmm), trans, obs_distributions(hmm), Geometric.(1 .- stay))
end

function rand_sequences(rng, model, lengths)
    obs_seq = reduce(vcat, [rand(rng, model, T).obs_seq for T in lengths])
    return obs_seq, cumsum(lengths)
end

function longest_segment(state_seq)
    segments = StateSegments(state_seq, 1, length(state_seq))
    return maximum(t_end - t_start + 1 for (t_start, t_end) in segments)
end

# Run `forward_backward!` in place, passing `kwargs` to the storage.
function forward_backward_storage(
    model,
    obs_seq,
    control_seq=Fill(nothing, length(obs_seq));
    seq_ends=(length(obs_seq),),
    kwargs...,
)
    storage = initialize_forward_backward(model, obs_seq, control_seq; seq_ends, kwargs...)
    forward_backward!(storage, model, obs_seq, control_seq; seq_ends)
    return storage
end

all_state_seqs(N, T) = (collect(s) for s in Iterators.product(ntuple(_ -> 1:N, T)...))

#= Enumerate every state sequence whose sojourns last at most `max_duration`, in `BigFloat`.
`α[:, t]` is the posterior of the last state given `obs_seq[1:t]`, and `log_prefix[t]` its
loglikelihood. The counts are those of completed and censored sojourns of each duration. =#
function brute_force(
    hsmm, obs_seq, control_seq=Fill(nothing, length(obs_seq)); max_duration=length(obs_seq)
)
    N, T = length(hsmm), length(obs_seq)
    α, γ = zeros(BigFloat, N, T), zeros(BigFloat, N, T)
    ξ = [zeros(BigFloat, N, N) for _ in 1:T]
    completed = zeros(BigFloat, min(max_duration, T), N)
    censored = zeros(BigFloat, min(max_duration, T), N)
    q, q_logL, runner_up_logL = zeros(Int, T), -Inf, -Inf
    for t in 1:T, s in all_state_seqs(N, t)
        longest_segment(s) <= max_duration || continue
        logL = joint_logdensityof(hsmm, obs_seq[1:t], s, control_seq[1:t])
        p = exp(big(logL))
        α[s[t], t] += p
        t == T || continue
        for u in 1:T
            γ[s[u], u] += p
            u < T && s[u] != s[u + 1] && (ξ[u][s[u], s[u + 1]] += p)
        end
        for (t_start, t_end) in StateSegments(s, 1, T)
            counts = t_end == T ? censored : completed
            counts[t_end - t_start + 1, s[t_start]] += p
        end
        if logL > q_logL
            q, q_logL, runner_up_logL = s, logL, q_logL
        elseif logL > runner_up_logL
            runner_up_logL = logL
        end
    end
    log_prefix = log.(vec(sum(α; dims=1)))
    Z = exp(log_prefix[T])
    return (;
        logL=Float64(log_prefix[T]),
        log_prefix=Float64.(log_prefix),
        α=Float64.(α ./ sum(α; dims=1)),
        γ=Float64.(γ ./ Z),
        ξ=[Float64.(ξₜ ./ Z) for ξₜ in ξ],
        completed=Float64.(completed ./ Z),
        censored=Float64.(censored ./ Z),
        q,
        q_logL,
        tie=q_logL ≈ runner_up_logL,
    )
end

function test_brute_force(
    hsmm, obs_seq, control_seq=Fill(nothing, length(obs_seq)); max_duration=length(obs_seq)
)
    bf = brute_force(hsmm, obs_seq, control_seq; max_duration)
    @test logdensityof(hsmm, obs_seq, control_seq; max_duration) ≈ bf.logL

    α, logL = forward(hsmm, obs_seq, control_seq; max_duration)
    @test α ≈ bf.α
    @test only(logL) ≈ bf.logL

    γ, logL = forward_backward(hsmm, obs_seq, control_seq; max_duration)
    @test γ ≈ bf.γ
    @test only(logL) ≈ bf.logL
    storage = forward_backward_storage(
        hsmm, obs_seq, control_seq; max_duration, duration_counts=true
    )
    @test storage.forward.log_prefix ≈ bf.log_prefix
    @test storage.γ ≈ bf.γ
    @test storage.ξ ≈ bf.ξ
    @test storage.completed_durations ≈ bf.completed atol = 1e-12
    @test storage.censored_durations ≈ bf.censored atol = 1e-12

    q, logL = viterbi(hsmm, obs_seq, control_seq; max_duration)
    @test only(logL) ≈ bf.q_logL
    @test only(logL) ≈ joint_logdensityof(hsmm, obs_seq, q, control_seq)
    @test longest_segment(q) <= max_duration
    @test bf.tie || q == bf.q
    return nothing
end
