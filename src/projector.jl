struct Projector{IndsT}
    data::Dict{IndsT,Int} # values must be positive.

    function Projector(data::Dict{IndsT,Int}) where IndsT
        for (k, v) in data
            if v <= 0
                error("Value at $(k) must be positive.")
            end
        end
        return new{IndsT}(data)
    end
end

Base.getindex(p::Projector, inds) = p.data[inds]

Base.:(==)(a::Projector, b::Projector)::Bool = (a.data == b.data)


"""
`a < b` means that `a` is projected at a subset of the indices that `b` is projected at.
"""
function Base.:(<)(a::Projector{IndsT}, b::Projector{IndsT})::Bool where IndsT
    for k in intersect(keys(a.data), keys(b.data))
        if a[k] != b[k]
            return false
        end
    end

    return keys(b.data) ⊆ keys(a.data)
end

Base.:(>)(a::Projector, b::Projector)::Bool = (b < a)

"""
`a & b` is the logical AND of the two projectors.
"""
function Base.:&(a::Projector{IndsT}, b::Projector{IndsT})::Projector{IndsT} where {IndsT}

end

function Base.:|(a::Projector, b::Projector)::Projector
end

Base.:<=(a::Projector, b::Projector)::Bool = (a < b) || (a == b)
Base.:>=(a::Projector, b::Projector) = (b <= a)

function hasoverlap(p1, p2)::Bool
    length(p1) == length(p2) || error("Length mismatch")
    for (a, b) in zip(Iterators.flatten(p1), Iterators.flatten(p2))
        if a != 0 && b != 0
            if a != b
                return false
            end
        end
    end
    return true
end

function isprojectedat(p::Projector{IndsT}, ind::IndsT)::Bool where {IndsT}
    return ind ∈ keys(p.data)
end