module HiddenMarkovModelsDistributionsExt

using ArgCheck: @argcheck
using HiddenMarkovModels: HiddenMarkovModels
using Distributions:
    Distributions,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    DiscreteUnivariateDistribution,
    Geometric,
    logccdf,
    fit,
    partype

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{<:UnivariateDistribution},
    i::Integer,
    x_nums::AbstractVector,
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), x_nums, w)
end

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{<:MultivariateDistribution},
    i::Integer,
    x_vecs::AbstractVector{<:AbstractVector},
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), reduce(hcat, x_vecs), w)
end

#=

# Matrix distribution fitting not supported by Distributions.jl at the moment

function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector{<:MatrixDistribution},
    i::Integer,
    x_mats::AbstractVector{<:AbstractMatrix},
    w::AbstractVector,
)
    return dists[i] = fit(typeof(dists[i]), reduce(dcat, x_mats), w)
end

dcat(M1, M2) = cat(M1, M2; dims=3)
=#

# Duration distributions model `sojourn - 1`, so `P(sojourn >= k) = P(dist > k - 2)`.
function HiddenMarkovModels.duration_logsurvival(
    dist::DiscreteUnivariateDistribution, k::Integer
)
    # `logccdf` can return `NaN` at the lower boundary.
    k <= one(k) && return zero(logccdf(dist, k))
    return logccdf(dist, k - 2)
end

# Closed form, see the formulas page.
function HiddenMarkovModels.fit_duration_in_sequence!(
    dists::AbstractVector{<:Geometric},
    i::Integer,
    completed::AbstractVector,
    censored::AbstractVector,
)
    @argcheck sum(completed) > 0
    ends = sum(completed)
    steps = sum(
        d * completed[d] + (d - 1) * censored[d] for d in eachindex(completed, censored)
    )
    T = partype(dists[i])
    q = convert(T, ends / steps)
    #= `q = 1` (all sojourns last one step) is the supremum, but `logpdf(Geometric(1), 0)` is NaN
    (`0 * log(0)`) in Distributions.jl, so stay one ulp below it. =#
    dists[i] = Geometric(min(q, prevfloat(one(T))))
    return nothing
end

end
