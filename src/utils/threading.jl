"""
$(SIGNATURES)

Call `f(k)` for every sequence index `k` in `seq_ends`, spreading sequences across tasks.

A single sequence given as an `NTuple{1}` runs serially, which keeps that path type-stable and
allocation-free.
"""
function foreach_sequence(f::F, seq_ends::AbstractVectorOrNTuple{Int}) where {F}
    scheduler = seq_ends isa NTuple{1} ? SerialScheduler() : DynamicScheduler()
    tforeach(f, eachindex(seq_ends); scheduler)
    return nothing
end
