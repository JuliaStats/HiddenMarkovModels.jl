"""
$(TYPEDEF)

Basic implementation of an HSMM with explicit state durations.

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
    "state transition probabilities (zero diagonal, no self-transitions)"
    trans::M
    "observation distributions"
    dists::VD
    "state duration distributions, interpreted as the law of `(sojourn - 1)`"
    dur_dists::VDur
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function HSMM(
        init::AbstractVector,
        trans::AbstractMatrix,
        dists::AbstractVector,
        dur_dists::AbstractVector,
    )
        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans)
        hsmm = new{
            typeof(init),
            typeof(trans),
            typeof(dists),
            typeof(dur_dists),
            typeof(log_init),
            typeof(log_trans),
        }(
            init, trans, dists, dur_dists, log_init, log_trans
        )
        @argcheck valid_hsmm(hsmm)
        return hsmm
    end
end

function Base.show(io::IO, hsmm::HSMM)
    return print(
        io,
        "Hidden Semi-Markov Model with:\n - initialization: $(hsmm.init)\n - transition matrix: $(hsmm.trans)\n - observation distributions: [$(join(hsmm.dists, ", "))]\n - duration distributions: [$(join(hsmm.dur_dists, ", "))]",
    )
end

initialization(hsmm::HSMM) = hsmm.init
log_initialization(hsmm::HSMM) = hsmm.loginit
transition_matrix(hsmm::HSMM) = hsmm.trans
log_transition_matrix(hsmm::HSMM) = hsmm.logtrans
obs_distributions(hsmm::HSMM) = hsmm.dists
duration_distributions(hsmm::HSMM) = hsmm.dur_dists
