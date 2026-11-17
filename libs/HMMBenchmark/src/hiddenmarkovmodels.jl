struct HiddenMarkovModelsImplem <: Implementation end
Base.string(::HiddenMarkovModelsImplem) = "HiddenMarkovModels.jl"

function build_model(::HiddenMarkovModelsImplem, instance::Instance, params::Params)
    (; custom_dist, nb_states, obs_dim, hsmm) = instance
    (; init, trans, means, stds, dur_means) = params

    if obs_dim == 1
        dists = [Normal(means[1, i], stds[1, i]) for i in 1:nb_states]
    elseif custom_dist
        dists = [LightDiagNormal(means[:, i], stds[:, i]) for i in 1:nb_states]
    else
        dists = [MvNormal(means[:, i], Diagonal(stds[:, i])) for i in 1:nb_states]
    end

    if hsmm
        return HiddenMarkovModels.HSMM(init, trans, dists, Poisson.(dur_means))
    else
        return HiddenMarkovModels.HMM(init, trans, dists)
    end
end

# The `max_duration` keyword for an HSMM, nothing for an HMM.
function duration_kwargs(instance::Instance)
    (; hsmm, max_duration) = instance
    return (hsmm && max_duration > 0) ? (; max_duration) : (;)
end

function _set_params!(b; samples, seconds)
    b.params.samples = samples
    b.params.seconds = seconds
    return b
end

function build_benchmarkables(
    implem::HiddenMarkovModelsImplem,
    instance::Instance,
    params::Params,
    data::AbstractArray{<:Real,3},
    algos::Vector{String};
    samples::Int=50,
    seconds::Real=5.0,
)
    (; obs_dim, seq_length, nb_seqs, bw_iter) = instance
    hmm = build_model(implem, instance, params)

    obs_seqs = if obs_dim == 1
        [[data[k, t, 1] for t in 1:seq_length] for k in 1:nb_seqs]
    else
        [[data[k, t, :] for t in 1:seq_length] for k in 1:nb_seqs]
    end
    obs_seq = reduce(vcat, obs_seqs)
    control_seq = fill(nothing, length(obs_seq))
    seq_ends = cumsum(length.(obs_seqs))

    benchs = Dict{String,Any}()
    set! = b -> _set_params!(b; samples=samples, seconds=seconds)
    kw = duration_kwargs(instance)

    allocating = (
        ("forward", forward), ("viterbi", viterbi), ("forward_backward", forward_backward)
    )
    for (name, fn) in allocating
        if name in algos
            benchs[name] = set!(
                @benchmarkable(
                    $fn($hmm, $obs_seq, $control_seq; seq_ends=($seq_ends), $(kw)...),
                    evals = 1,
                )
            )
        end
    end

    inplace = (
        ("forward!", forward!, initialize_forward),
        ("viterbi!", viterbi!, initialize_viterbi),
        ("forward_backward!", forward_backward!, initialize_forward_backward),
    )
    for (name, fn, init) in inplace
        if name in algos
            benchs[name] = set!(
                @benchmarkable(
                    $fn(storage, $hmm, $obs_seq, $control_seq; seq_ends=($seq_ends)),
                    evals = 1,
                    setup = (
                        storage = $init(
                            $hmm, $obs_seq, $control_seq; seq_ends=($seq_ends), $(kw)...
                        )
                    )
                )
            )
        end
    end

    if "baum_welch" in algos
        benchs["baum_welch"] = set!(
            @benchmarkable(
                baum_welch(
                    $hmm,
                    $obs_seq,
                    $control_seq;
                    seq_ends=($seq_ends),
                    max_iterations=($bw_iter),
                    atol=(-Inf),
                    loglikelihood_increasing=false,
                    $(kw)...,
                ),
                evals = 1,
            )
        )
    end
    if "baum_welch!" in algos
        benchs["baum_welch!"] = set!(
            @benchmarkable(
                baum_welch!(
                    fb_storage,
                    logL_evolution,
                    hmm_guess,
                    $obs_seq,
                    $control_seq;
                    seq_ends=($seq_ends),
                    max_iterations=($bw_iter),
                    atol=(-Inf),
                    loglikelihood_increasing=false,
                ),
                evals = 1,
                setup = (
                    hmm_guess = build_model($implem, $instance, $params);
                    fb_storage = initialize_baum_welch(
                        hmm_guess,
                        $obs_seq,
                        $control_seq;
                        seq_ends=($seq_ends),
                        max_duration=$(get(kw, :max_duration, nothing)),
                    );
                    logL_evolution = Float64[];
                    sizehint!(logL_evolution, $bw_iter)
                )
            )
        )
    end

    return benchs
end
