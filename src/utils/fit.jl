"""
$(SIGNATURES)

Modify the `i`-th element of `dists` by fitting it to an observation sequence `x` with associated weight sequence `w`.

Default behavior:

    fit!(dists[i], x, w)

Override for Distributions.jl (in the package extension)

    dists[i] = fit(eltype(dists), turn_into_vector(x), w)
"""
function fit_in_sequence!(dists::AbstractVector, i::Integer, x, w)
    fit!(dists[i], x, w)
    return nothing
end

#=
 For ControlledHMM, we need to pass control_seq to fit!
=#
function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector, i::Integer, x, control_seq, w
)
    StatsAPI.fit!(dists[i], x, control_seq, w)
    return nothing
end
