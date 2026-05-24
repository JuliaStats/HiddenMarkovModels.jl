"""
    valid_hsmm(hsmm, control=nothing)

Perform some checks to rule out obvious inconsistencies with an [`AbstractHSMM`](@ref) object.

On top of the HMM-style checks (probability initialization, stochastic transition matrix,
matching dimensions, valid observation distributions), this verifies:

- the transition matrix has a zero diagonal (no self-transitions);
- there is one duration distribution per state, and each carries a density.
"""
function valid_hsmm(hsmm::AbstractHSMM, control=nothing)
    _valid_state_space_core(hsmm, control) || return false

    trans = transition_matrix(hsmm, control)
    durations = duration_distributions(hsmm, control)
    N = length(hsmm)

    # One duration distribution per state.
    length(durations) == N || return false

    # No self-transitions (allowing for numerical noise).
    for i in 1:N
        if trans[i, i] > 1e-10
            return false
        end
    end

    # Duration distributions must carry a density.
    for dur_dist in durations
        if DensityKind(dur_dist) == NoDensity()
            return false
        end
    end

    return true
end
