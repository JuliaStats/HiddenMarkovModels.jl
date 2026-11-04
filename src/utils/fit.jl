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

#=
Sum the transition marginals `ξ[t1:t2-1]` of each sequence into `ξ[t2]`, which is free to use
as scratch space since no transition leaves the last timestep (its marginal is zero).
=#
function sum_transition_marginals!(
    ξ::AbstractVector{<:AbstractMatrix}, seq_ends::AbstractVectorOrNTuple{Int}
)
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            sum_transition_marginals!(ξ, seq_ends, k)
        end
    else
        @threads for k in eachindex(seq_ends)
            sum_transition_marginals!(ξ, seq_ends, k)
        end
    end
    return nothing
end

function sum_transition_marginals!(
    ξ::AbstractVector{<:AbstractMatrix}, seq_ends::AbstractVectorOrNTuple{Int}, k::Integer
)
    t1, t2 = seq_limits(seq_ends, k)
    # An empty sequence has no slot of its own: `ξ[t2]` belongs to the previous sequence.
    t1 > t2 && return nothing
    scratch = ξ[t2]
    fill!(scratch, zero(eltype(scratch)))
    for t in t1:(t2 - 1)
        scratch .+= ξ[t]
    end
    return nothing
end

function first_nonempty_sequence(seq_ends::AbstractVectorOrNTuple{Int})
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        t1 <= t2 && return k
    end
    throw(ArgumentError("At least one sequence must be nonempty."))
end

#=
Re-estimate `trans` from the transition marginals, summed over the nonempty sequences in the
scratch slot `ξ[t2]` of the first one, by normalizing each row. A row without outgoing mass would
give `0 / 0`, so it keeps its previous values. `ξ` no longer holds the marginals afterwards.
Nothing is written before the check that some sequence is nonempty, so call this first in `fit!`.
=#
function fit_transitions!(
    trans::AbstractMatrix,
    ξ::AbstractVector{<:AbstractMatrix},
    seq_ends::AbstractVectorOrNTuple{Int},
)
    k_first = first_nonempty_sequence(seq_ends)
    sum_transition_marginals!(ξ, seq_ends)
    _, t2_first = seq_limits(seq_ends, k_first)
    total = ξ[t2_first]
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        (k == k_first || t1 > t2) && continue
        total .+= ξ[t2]
    end
    normalize_transitions!(trans, total)
    return nothing
end

function normalize_transitions!(trans::AbstractMatrix, total::AbstractMatrix)
    for i in axes(trans, 1)
        outgoing = zero(eltype(total))
        for j in axes(trans, 2)
            outgoing += total[i, j]
        end
        iszero(outgoing) && continue
        for j in axes(trans, 2)
            trans[i, j] = total[i, j] / outgoing
        end
    end
    return nothing
end

#= Scalar indexing of a sparse matrix is a binary search, so work on the stored entries, which
`total` shares with `trans` when it was allocated with `similar`. =#
function normalize_transitions!(trans::SparseMatrixCSC, total::SparseMatrixCSC)
    if trans.colptr != total.colptr || rowvals(trans) != rowvals(total)
        return invoke(
            normalize_transitions!, Tuple{AbstractMatrix,AbstractMatrix}, trans, total
        )
    end
    rows, vals, total_vals = rowvals(trans), nonzeros(trans), nonzeros(total)
    outgoing = zeros(eltype(total), size(trans, 1))
    for p in eachindex(total_vals)
        outgoing[rows[p]] += total_vals[p]
    end
    for p in eachindex(vals)
        i = rows[p]
        iszero(outgoing[i]) || (vals[p] = total_vals[p] / outgoing[i])
    end
    return nothing
end
