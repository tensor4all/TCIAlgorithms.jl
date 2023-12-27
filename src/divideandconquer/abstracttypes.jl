# Type for an object that can be projected on a subset of indices
abstract type ProjectableEvaluator{T} <: TCI.BatchEvaluator{T} end

function iscompatible(obj::ProjectableEvaluator{T}, indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::Bool where {T}
    error("Must be implemented!")
end

function projector(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.projector[l][ilegg] for l in 1:length(obj)]
end

function sitedims(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.sitedims[l][ilegg] for l in 1:length(obj)]
end


struct Projector
    data::Vector{Vector{Int}}
end

function Base.iterate(p::Projector, state=1)
    if state > length(p.data)
        return nothing
    end
    return (p.data[state], state + 1)
end

Base.length(p::Projector) = length(p.data)