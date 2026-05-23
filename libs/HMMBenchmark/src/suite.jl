function to_namedtuple(x)
    return NamedTuple(n => getfield(x, n) for n in fieldnames(typeof(x)))
end

function define_suite(
    rng::AbstractRNG,
    implems::Vector{<:Implementation}=[HiddenMarkovModelsImplem()];
    instances::Vector{<:Instance},
    algos::Vector{String},
)
    SUITE = BenchmarkGroup()
    for implem in implems
        for instance in instances
            params = build_params(rng, instance)
            data = build_data(rng, instance)
            bench_tup = build_benchmarkables(implem, instance, params, data, algos)
            for (algo, bench) in pairs(bench_tup)
                SUITE[string(implem)][string(instance)][algo] = bench
            end
        end
    end
    return SUITE
end

function parse_results(
    results;
    path=nothing,
    aggregators=[
        :minimum => minimum,
        :median => median,
        :maximum => maximum,
        :mean => mean,
        :std => std,
        :q25 => x -> quantile(x, 0.25),
        :q75 => x -> quantile(x, 0.75),
    ],
)
    data = DataFrame()
    for implem_str in identity.(keys(results))
        for instance_str in identity.(keys(results[implem_str]))
            instance = Instance(instance_str)
            for algo in identity.(keys(results[implem_str][instance_str]))
                perf = results[implem_str][instance_str][algo]
                perf_dict = Dict{Symbol,Number}()
                perf_dict[:samples] = length(perf.times)
                for (name, agg) in aggregators
                    perf_dict[Symbol("time_$name")] = agg(perf.times)
                end
                row = merge((; implem=implem_str, algo), to_namedtuple(instance))
                row = merge(row, perf_dict)
                push!(data, row)
            end
        end
    end

    if !isnothing(path)
        CSV.write(path, data)
    end
    return data
end

function read_results(path)
    return CSV.read(path, DataFrame)
end
