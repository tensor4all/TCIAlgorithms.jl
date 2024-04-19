"""
Type for an object that can be projected on a subset of indices

Attributes:
- projector: Projector object
- sitedims: Vector{Vector{Int}} of the dimensions of the local indices
"""
abstract type ProjectableEvaluator{T} <: TCI.BatchEvaluator{T} end

struct Projector
    data::Vector{Vector{Int}} # 0 means no projection
    sitedims::Vector{Vector{Int}}
    function Projector(data, sitedims)
        for (d, s) in zip(data, sitedims)
            length(d) == length(s) || error("Length mismatch")
            for (d_, s_) in zip(d, s)
                if d_ > s_ || d_ < 0
                    error("Invalid projector")
                end
            end
        end
        return new(data, sitedims)
    end
end

function sum(obj::ProjectableEvaluator{T})::T where {T}
    error("Must be implemented!")
    return zero(T)
end

function project(
    obj::ProjectableEvaluator{T}, prj::Projector
)::ProjectableEvaluator{T} where {T}
    return error("Must be implemented!")
end

function projector(obj::ProjectableEvaluator{T})::Projector where {T}
    return obj.projector
end

function projector(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.projector[l][ilegg] for l in 1:length(obj)]
end

function sitedims(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.sitedims[l][ilegg] for l in 1:length(obj)]
end

function Base.copy(obj::Projector)
    return Projector(deepcopy(obj.data))
end

function Base.iterate(p::Projector, state=1)
    if state > length(p.data)
        return nothing
    end
    return (p.data[state], state + 1)
end

Base.length(p::Projector) = length(p.data)
Base.getindex(p::Projector, index::Int) = p.data[index]

function (p::Projector)(isite::Int, ilegg::Int)
    return p.data[isite][ilegg]
end

function only(p::Projector, ilegg::Int)::Projector
    return Projector([[p.data[l][ilegg]] for l in 1:length(p)])
end

Base.:(==)(a::Projector, b::Projector)::Bool = (a.data == b.data)
Base.:(<)(a::Projector, b::Projector)::Bool = (a <= b) && (a != b)
Base.:(>)(a::Projector, b::Projector)::Bool = b < a

function Base.:&(a::Projector, b::Projector)::Projector
    a.sitedims == b.sitedims || error("Sitedims mismatch")
    length(a) == length(b) || error("Length mismatch")
    ab = Vector{Int}[]
    for (a_, b_) in zip(a, b)
        ab_ = Int[]
        for (a__, b__) in zip(a_, b_)
            if a__ == 0
                push!(ab_, b__)
            elseif b__ == 0
                push!(ab_, a__)
            elseif a__ == b__
                push!(ab_, a__)
            else
                error("Incompatible projectors $(a) && $(b)")
            end
        end
        push!(ab, ab_)
    end

    return Projector(ab, a.sitedims)
end

function Base.:<=(a::Projector, b::Projector)::Bool
    length(a) == length(b) || error("Length mismatch")
    length(a) == length(b) || error("Length mismatch")
    for (a_, b_) in zip(Iterators.flatten(a), Iterators.flatten(b))
        if a_ != 0 && b_ != 0
            if a_ != b_
                return false
            end
        elseif a_ == 0
            if b_ != 0
                return false
            end
        elseif b_ == 0
            # Everything is fine
        end
    end
    return true
end

Base.:>=(a::Projector, b::Projector) = (b <= a)

Base.:<=(a::Vector{Vector{Int}}, b::Projector) = (Projector(a, b.sitedims) <= b)

function hasoverlap(p1::Projector, p2::Projector)::Bool
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

function leftindexset_contained(p1::Projector, p2::Projector)::Bool
    for (a, b) in zip(Iterators.flatten(p1), Iterators.flatten(p2))
        if a != 0 && b != 0
            if a != b
                return false
            end
        elseif a == 0 && b != 0
            return false
        end
    end
    return true
end

function Base.reshape(projector::Projector, dims::AbstractVector{<:AbstractVector{Int}})::Projector
    Projector(
        [mulltii(newdim, lineari(olddim, p)) for (p, olddim, newdim) in zip(projector.data, obj.sitedims, dims)]
        )
end