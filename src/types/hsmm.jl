"""
$(TYPEDEF)

Basic implementation of a Hidden Semi-Markov Model.

Self-transitions are removed from `trans` at construction (the diagonal is zeroed and each row is
renormalized over the remaining off-diagonal entries), since sojourn lengths are governed by the
per-state duration distributions in `durations` rather than by repeated self-transitions.

# Fields

$(TYPEDFIELDS)
"""
struct HSMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    VDur<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHSMM
    "initial state probabilities"
    init::V
    "state transition probabilities (zero diagonal — no self-transitions)"
    trans::M
    "observation distributions, one per state"
    dists::VD
    "state duration distributions, one per state"
    durations::VDur
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function HSMM(
        init::AbstractVector,
        trans::AbstractMatrix,
        dists::AbstractVector,
        durations::AbstractVector,
    )
        trans_no_self = copy(trans)
        for i in axes(trans_no_self, 1)
            trans_no_self[i, i] = zero(eltype(trans_no_self))
        end
        foreach(sum_to_one!, eachrow(trans_no_self))

        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans_no_self)

        hsmm = new{
            typeof(init),
            typeof(trans_no_self),
            typeof(dists),
            typeof(durations),
            typeof(log_init),
            typeof(log_trans),
        }(
            init, trans_no_self, dists, durations, log_init, log_trans
        )
        @argcheck valid_hsmm(hsmm)
        return hsmm
    end
end

function Base.show(io::IO, hsmm::HSMM)
    return print(
        io,
        "Hidden Semi-Markov Model with:\n - initialization: $(hsmm.init)\n - transition matrix: $(hsmm.trans)\n - observation distributions: [$(join(hsmm.dists, ", "))]\n - duration distributions: [$(join(hsmm.durations, ", "))]",
    )
end

initialization(hsmm::HSMM) = hsmm.init
log_initialization(hsmm::HSMM) = hsmm.loginit
transition_matrix(hsmm::HSMM) = hsmm.trans
log_transition_matrix(hsmm::HSMM) = hsmm.logtrans
obs_distributions(hsmm::HSMM) = hsmm.dists
duration_distributions(hsmm::HSMM) = hsmm.durations

## Fitting

function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ, η) = fb_storage
    N = length(hsmm)

    # Aggregate transition marginals for each sequence into ξ[t2] as scratch.
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    end

    # Re-estimate init and transition matrix.
    fill!(hsmm.init, zero(eltype(hsmm.init)))
    fill!(hsmm.trans, zero(eltype(hsmm.trans)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hsmm.init .+= view(γ, :, t1)
        hsmm.trans .+= ξ[t2]
    end
    sum_to_one!(hsmm.init)

    # Diagonal of trans is structurally zero (no self-transitions); ξ never populates it,
    # so each row sum is over off-diagonal entries only.
    for i in 1:N
        row = view(hsmm.trans, i, :)
        s = sum(row)
        if s > zero(s)
            row ./= s
        else
            # No outgoing transitions observed for state i — fall back to uniform off-diagonal
            # so the row stays a valid probability vector.
            fill!(row, one(eltype(row)) / (N - 1))
            hsmm.trans[i, i] = zero(eltype(hsmm.trans))
        end
    end

    # Refit observation distributions.
    for i in 1:N
        fit_in_sequence!(hsmm.dists, i, obs_seq, view(γ, i, :))
    end

    # Refit duration distributions from expected segment-duration counts.
    duration_support = collect(1:fb_storage.max_duration)
    for i in 1:N
        weights = view(η, :, i)
        if sum(weights) > zero(eltype(weights))
            fit!(hsmm.durations[i], duration_support, weights)
        end
    end

    hsmm.loginit .= log.(hsmm.init)
    hsmm.logtrans .= log.(hsmm.trans)

    @argcheck valid_hsmm(hsmm)
    return nothing
end
