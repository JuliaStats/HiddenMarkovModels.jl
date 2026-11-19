using DensityInterface: DensityInterface, DensityKind, HasDensity, logdensityof
using HiddenMarkovModels
using HiddenMarkovModels:
    AbstractLatentStateModel,
    AbstractHSMM,
    HSMMForwardBackwardStorage,
    HSMMForwardStorage,
    HSMMViterbiStorage,
    duration_distributions,
    duration_logdensityof,
    duration_logsurvival,
    duration_logdensity_type,
    elementwise_log,
    Fill,
    StateSegments,
    fit_duration_in_sequence!,
    forward!,
    forward_backward!,
    initialize_forward,
    initialize_forward_backward,
    initialize_viterbi,
    log_initialization,
    log_transition_matrix,
    valid_hsmm,
    viterbi!
using Distributions: Binomial, Categorical, Geometric, Normal, Poisson, mean, std, succprob
using FiniteDifferences: FiniteDifferences, central_fdm
using ForwardDiff: ForwardDiff
using JET: @test_call, @test_opt
using Random: AbstractRNG
using SparseArrays: SparseMatrixCSC, nnz, sparse
using StableRNGs: StableRNG
using Test
using Zygote: Zygote

# A duck-typed equivalent of `Geometric(p)`.
struct DuckGeometric
    p::Float64
end
DensityInterface.DensityKind(::DuckGeometric) = HasDensity()
function DensityInterface.logdensityof(d::DuckGeometric, k)
    return k < 0 ? -Inf : log(d.p) + k * log1p(-d.p)
end
Base.rand(::AbstractRNG, ::DuckGeometric) = 0
function HiddenMarkovModels.duration_logsurvival(d::DuckGeometric, k::Integer)
    return k <= 1 ? 0.0 : (k - 1) * log1p(-d.p)
end

# The control at the start of a segment selects its duration distribution.
struct ControlledDurationHSMM <: AbstractHSMM
    init::Vector{Float64}
    trans::Matrix{Float64}
    dists::Vector{Normal{Float64}}
    durs::Vector{Vector{Geometric{Float64}}}
end

HiddenMarkovModels.initialization(hsmm::ControlledDurationHSMM) = hsmm.init
HiddenMarkovModels.transition_matrix(hsmm::ControlledDurationHSMM) = hsmm.trans
HiddenMarkovModels.transition_matrix(hsmm::ControlledDurationHSMM, ::Integer) = hsmm.trans
HiddenMarkovModels.obs_distributions(hsmm::ControlledDurationHSMM) = hsmm.dists
HiddenMarkovModels.obs_distributions(hsmm::ControlledDurationHSMM, ::Integer) = hsmm.dists
HiddenMarkovModels.duration_distributions(hsmm::ControlledDurationHSMM) = hsmm.durs[1]
function HiddenMarkovModels.duration_distributions(
    hsmm::ControlledDurationHSMM, control::Integer
)
    return hsmm.durs[control]
end

# Every component depends on the control, so using the wrong control changes the results.
struct FullyControlledHSMM <: AbstractHSMM
    init::Vector{Float64}
    trans::Vector{Matrix{Float64}}
    dists::Vector{Vector{Normal{Float64}}}
    durs::Vector{Vector{Geometric{Float64}}}
end

HiddenMarkovModels.initialization(hsmm::FullyControlledHSMM) = hsmm.init
HiddenMarkovModels.transition_matrix(hsmm::FullyControlledHSMM) = hsmm.trans[1]
function HiddenMarkovModels.transition_matrix(hsmm::FullyControlledHSMM, control::Integer)
    return hsmm.trans[control]
end
HiddenMarkovModels.obs_distributions(hsmm::FullyControlledHSMM) = hsmm.dists[1]
function HiddenMarkovModels.obs_distributions(hsmm::FullyControlledHSMM, control::Integer)
    return hsmm.dists[control]
end
HiddenMarkovModels.duration_distributions(hsmm::FullyControlledHSMM) = hsmm.durs[1]
function HiddenMarkovModels.duration_distributions(
    hsmm::FullyControlledHSMM, control::Integer
)
    return hsmm.durs[control]
end

function fully_controlled_hsmm()
    init = [0.5, 0.3, 0.2]
    trans = [
        [0.0 0.9 0.1; 0.2 0.0 0.8; 0.6 0.4 0.0], [0.0 0.1 0.9; 0.8 0.0 0.2; 0.3 0.7 0.0]
    ]
    dists = [
        [Normal(0.0), Normal(2.0), Normal(4.0)], [Normal(4.0), Normal(0.0), Normal(2.0)]
    ]
    durs = [
        [Geometric(0.2), Geometric(0.5), Geometric(0.8)],
        [Geometric(0.7), Geometric(0.3), Geometric(0.4)],
    ]
    return FullyControlledHSMM(init, trans, dists, durs)
end

#= Sojourns last at most 2 timesteps and state 2 almost never emits `1`, so a self-transition of
probability `1e-17` in state 1 would explain `[1, 1, 1, 1]` better than the valid paths. =#
const SELF_TRANSITION_OBS = [1, 1, 1, 1]
function self_transition_hsmm(trans)
    dists = [Categorical([1.0, 0.0]), Categorical([1e-20, 1 - 1e-20])]
    return HSMM([0.6, 0.4], trans, dists, [Binomial(1, 0.5), Binomial(1, 0.5)])
end
const NEAR_ZERO_DIAGONALS = ([1e-17 1.0; 1.0 1e-17], sparse([1e-17 1.0; 1.0 1e-17]))

all_state_seqs(N, T) = (collect(s) for s in Iterators.product(ntuple(_ -> 1:N, T)...))

function brute_force_logdensityof(hsmm, obs_seq, control_seq, N)
    T = length(obs_seq)
    return Float64(
        log(
            sum(
                exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq))) for
                s in all_state_seqs(N, T)
            ),
        ),
    )
end

function brute_force_marginals(hsmm, obs_seq, control_seq, N)
    T = length(obs_seq)
    α = zeros(BigFloat, N, T)
    for t in 1:T
        for s in all_state_seqs(N, t)
            α[s[t], t] += exp(
                big(joint_logdensityof(hsmm, obs_seq[1:t], s, control_seq[1:t]))
            )
        end
        α[:, t] ./= sum(α[:, t])
    end
    return Float64.(α)
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

function rand_hsmm(rng, N)
    init = rand(rng, N)
    init ./= sum(init)
    trans = [i == j ? 0.0 : rand(rng) for i in 1:N, j in 1:N]
    for i in 1:N
        trans[i, :] ./= sum(trans[i, :])
    end
    dists = [Normal(2.0 * i, 1.0) for i in 1:N]
    dur_dists = [Geometric(0.2 + 0.15 * i) for i in 1:N]
    return HSMM(init, trans, dists, dur_dists)
end

function longest_segment(state_seq)
    return maximum(
        t_end - t_start + 1 for
        (t_start, t_end) in StateSegments(state_seq, 1, length(state_seq))
    )
end

function brute_force_posteriors(hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq))
    T = length(obs_seq)
    γ = zeros(BigFloat, N, T)
    ξ = [zeros(BigFloat, N, N) for _ in 1:(T - 1)]
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        p = exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq)))
        for t in 1:T
            γ[s[t], t] += p
        end
        for t in 1:(T - 1)
            if s[t] != s[t + 1]
                ξ[t][s[t], s[t + 1]] += p
            end
        end
    end
    Z = sum(γ[:, 1])
    return Float64.(γ ./ Z), [Float64.(ξₜ ./ Z) for ξₜ in ξ], Float64(log(Z))
end

function full_forward_backward(
    hsmm, obs_seq, control_seq=fill(nothing, length(obs_seq)); kwargs...
)
    seq_ends = get(kwargs, :seq_ends, (length(obs_seq),))
    storage = initialize_forward_backward(hsmm, obs_seq, control_seq; kwargs...)
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    return storage
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward_backward!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

# Unlike the marginals, maximization never sums probabilities, so `Float64` is precise enough.
function brute_force_viterbi(hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq))
    T = length(obs_seq)
    best_q, best_logL = zeros(Int, T), -Inf
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        logL = joint_logdensityof(hsmm, obs_seq, s, control_seq)
        if logL > best_logL
            best_q, best_logL = s, logL
        end
    end
    return best_q, best_logL
end

# Ensure `@allocated` sees concretely typed arguments.
function call_viterbi!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return viterbi!(storage, hsmm, obs_seq, control_seq; seq_ends)
end

#= Expected numbers of completed and right-censored sojourns of each duration, by summing the
posterior weight of every state sequence over its segments. =#
function brute_force_duration_counts(
    hsmm, obs_seq, control_seq, N; max_duration=length(obs_seq)
)
    T = length(obs_seq)
    D = min(max_duration, T)
    completed = zeros(BigFloat, D, N)
    censored = zeros(BigFloat, D, N)
    Z = big(0.0)
    for s in all_state_seqs(N, T)
        longest_segment(s) <= max_duration || continue
        p = exp(big(joint_logdensityof(hsmm, obs_seq, s, control_seq)))
        Z += p
        for (t_start, t_end) in StateSegments(s, 1, T)
            counts = t_end == T ? censored : completed
            counts[t_end - t_start + 1, s[t_start]] += p
        end
    end
    return Float64.(completed ./ Z), Float64.(censored ./ Z)
end

function full_forward_backward_counts(
    hsmm, obs_seq, control_seq=fill(nothing, length(obs_seq)); kwargs...
)
    seq_ends = get(kwargs, :seq_ends, (length(obs_seq),))
    storage = initialize_forward_backward(
        hsmm, obs_seq, control_seq; duration_counts=true, kwargs...
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true)
    return storage
end

# Ensure `@allocated` sees concretely typed arguments.
function call_forward_backward_counts!(storage, hsmm, obs_seq, control_seq, seq_ends)
    return forward_backward!(
        storage, hsmm, obs_seq, control_seq; seq_ends, duration_counts=true
    )
end

function rand_sequences(rng, model, lengths)
    obs_seq = reduce(vcat, [rand(rng, model, T).obs_seq for T in lengths])
    return obs_seq, cumsum(lengths)
end

# Ensure `@allocated` sees concretely typed arguments.
function call_fit!(hsmm, storage, obs_seq, control_seq, seq_ends)
    return fit!(hsmm, storage, obs_seq, control_seq; seq_ends)
end

# Two states, and sequences of very different lengths, including one of length 1.
function two_state_problem()
    hsmm = HSMM(
        [0.6, 0.4],
        [0.0 1.0; 1.0 0.0],
        [Normal(0.0), Normal(3.0)],
        [Geometric(0.3), Geometric(0.5)],
    )
    obs_seq, seq_ends = rand_sequences(StableRNG(507), hsmm, [30, 1, 45, 20])
    return hsmm, obs_seq, seq_ends
end
