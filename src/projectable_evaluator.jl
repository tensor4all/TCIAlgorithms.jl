"""
Represents an object that can be projected on a subset of indices


Attributes:
- projector: Projector object
- sitedims: Vector{Vector{Int}} of the dimensions of the local indices
"""
abstract type ProjectableEvaluator{T} <: TCI.BatchEvaluator{T} end

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

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(indexset::MultiIndex)::T where {T}
    return obj(multii(obj.sitedims, indexset))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [multii(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [multii(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]

    return obj(leftindexset_, rightindexset_, Val(M))
end
