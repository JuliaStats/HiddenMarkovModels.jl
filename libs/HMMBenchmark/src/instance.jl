Base.@kwdef struct Instance
    nb_states::Int
    obs_dim::Int
    seq_length::Int = 100
    nb_seqs::Int = 20
    bw_iter::Int = 1
    sparse::Bool = false
    custom_dist::Bool = false
end

function Base.string(c::Instance)
    return join(("$n=$(getfield(c, n))" for n in fieldnames(typeof(c))), ";")
end

function Instance(s::AbstractString)
    kwargs = Dict{Symbol,Any}()
    for entry in split(s, ";")
        k, v = split(entry, "="; limit=2)
        sym = Symbol(k)
        kwargs[sym] = parse(fieldtype(Instance, sym), v)
    end
    return Instance(; kwargs...)
end

function build_data(rng::AbstractRNG, instance::Instance)
    (; nb_seqs, seq_length, obs_dim) = instance
    return randn(rng, nb_seqs, seq_length, obs_dim)
end
