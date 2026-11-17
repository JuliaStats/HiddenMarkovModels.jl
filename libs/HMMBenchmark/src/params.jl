Base.@kwdef struct Params{T,M<:AbstractMatrix{T}}
    init::Vector{T}
    trans::M
    means::Matrix{T}
    stds::Matrix{T}
    dur_means::Vector{T}  # HSMM only
end

function build_params(rng::AbstractRNG, instance::Instance)
    (; sparse, nb_states, obs_dim, hsmm) = instance
    init = ones(nb_states) / nb_states
    if sparse
        trans = spdiagm(
            0 => rand(rng, nb_states) / 2,
            +1 => rand(rng, nb_states - 1) / 2,
            -(nb_states - 1) => rand(rng, 1) / 2,
        )
    else
        trans = rand(rng, nb_states, nb_states)
    end
    if hsmm  # the sojourns are given by the durations
        for i in 1:nb_states
            trans[i, i] = 0
        end
    end
    for row in eachrow(trans)
        row ./= sum(row)
    end
    means = randn(rng, obs_dim, nb_states)
    stds = ones(obs_dim, nb_states)
    # Drawn last and only for HSMMs, so that the HMM instances keep their parameters.
    dur_means = hsmm ? 10 * rand(rng, nb_states) : Float64[]
    return Params(; init, trans, means, stds, dur_means)
end
