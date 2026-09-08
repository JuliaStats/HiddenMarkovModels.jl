"""
    StateSegments(state_seq, t1, t2)

Non-allocating iterator over the maximal constant-state runs of `state_seq[t1:t2]`.

Each element is a tuple `(t_start, t_end)` such that `state_seq[t_start:t_end]` is constant and
differs from its neighbors within `t1:t2`. Consecutive segments tile `t1:t2` in order.
"""
struct StateSegments{V<:AbstractVector}
    state_seq::V
    t1::Int
    t2::Int
end

Base.IteratorSize(::Type{<:StateSegments}) = Base.SizeUnknown()
Base.eltype(::Type{<:StateSegments}) = Tuple{Int,Int}

function Base.iterate(segments::StateSegments, t_start::Int=segments.t1)
    (; state_seq, t2) = segments
    t_start > t2 && return nothing
    t_end = t_start
    while t_end < t2 && state_seq[t_end + 1] == state_seq[t_start]
        t_end += 1
    end
    return (t_start, t_end), t_end + 1
end
