struct dynamaxImplem <: Implementation end
Base.string(::dynamaxImplem) = "dynamax"

function HMMBenchmark.build_model(implem::dynamaxImplem, instance::Instance, params::Params)
    jax = pyimport("jax")
    jnp = pyimport("jax.numpy")
    dynamax_hmm = pyimport("dynamax.hidden_markov_model")
    jax.config.update("jax_enable_x64", true)

    (; nb_states, obs_dim) = instance
    (; init, trans, means, stds) = params

    initial_probs = jnp.array(Py(init).to_numpy())
    transition_matrix = jnp.array(Py(trans).to_numpy())
    emission_means = jnp.array(Py(transpose(means)).to_numpy())
    emission_scale_diags = jnp.array(Py(transpose(stds)).to_numpy())

    hmm = dynamax_hmm.DiagonalGaussianHMM(nb_states, obs_dim)
    dyn_params, dyn_props = hmm.initialize(;
        initial_probs=initial_probs,
        transition_matrix=transition_matrix,
        emission_means=emission_means,
        emission_scale_diags=emission_scale_diags,
    )

    return hmm, dyn_params, dyn_props
end

function HMMBenchmark.build_benchmarkables(
    implem::dynamaxImplem,
    instance::Instance,
    params::Params,
    data::AbstractArray{<:Real,3},
    algos::Vector{String},
)
    jax = pyimport("jax")
    jnp = pyimport("jax.numpy")
    jax.config.update("jax_enable_x64", true)

    (; bw_iter) = instance
    hmm, dyn_params, _ = build_model(implem, instance, params)

    obs_tens_jax_py = jnp.array(Py(data).to_numpy())

    benchs = Dict()

    if "forward" in algos
        filter_vmap = jax.jit(jax.vmap(hmm.filter; in_axes=pylist((pybuiltins.None, 0))))
        benchs["forward"] = @benchmarkable begin
            $(filter_vmap)($dyn_params, $obs_tens_jax_py)
        end evals = 1 samples = 20
    end

    if "viterbi" in algos
        most_likely_states_vmap = jax.jit(
            jax.vmap(hmm.most_likely_states; in_axes=pylist((pybuiltins.None, 0)))
        )
        benchs["viterbi"] = @benchmarkable begin
            $(most_likely_states_vmap)($dyn_params, $obs_tens_jax_py)
        end evals = 1 samples = 20
    end

    if "forward_backward" in algos
        smoother_vmap = jax.jit(
            jax.vmap(hmm.smoother; in_axes=pylist((pybuiltins.None, 0)))
        )
        benchs["forward_backward"] = @benchmarkable begin
            $(smoother_vmap)($dyn_params, $obs_tens_jax_py)
        end evals = 1 samples = 20
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = @benchmarkable begin
            hmm_guess.fit_em(
                dyn_params_guess,
                dyn_props_guess,
                $obs_tens_jax_py;
                num_iters=($bw_iter),
                verbose=false,
            )
        end evals = 1 samples = 20 setup = (
            tup = build_model($implem, $instance, $params);
            hmm_guess = tup[1];
            dyn_params_guess = tup[2];
            dyn_props_guess = tup[3]
        )
    end

    return benchs
end
