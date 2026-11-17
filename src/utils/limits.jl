"""
$(SIGNATURES)

Return a tuple `(t1, t2)` giving the begin and end indices of subsequence `k` within a set of sequences ending at `seq_ends`.
"""
function seq_limits(seq_ends::AbstractVectorOrNTuple{Int}, k::Integer)
    if k == 1
        return 1, seq_ends[k]
    else
        return seq_ends[k - 1] + 1, seq_ends[k]
    end
end

function check_seq_ends(seq_ends::AbstractVectorOrNTuple{Int})
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        @argcheck t1 <= t2 "Sequence $k is empty, remove it from `seq_ends`."
    end
    return nothing
end
