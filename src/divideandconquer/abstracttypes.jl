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