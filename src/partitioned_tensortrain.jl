# General localset is not supported!
struct PartitionedTensorTrain{T,N} <: TCI.BatchEvaluator{T}
    data::TensorTrain{T,N}
    projector::Vector{Vector{Int}}
end

Base.length(obj::PartitionedTensorTrain{T,N}) where {T,N} = length(obj.data)

function iscompatible(obj::PartitionedTensorTrain{T,N},
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::Bool where {T,N}
    @show obj.projector
    @show indexsets
    for n in 1:N-2
        if !iscompatible(obj.projector[n], [i[n] for i in indexsets])
            return false
        end
    end
    return true
end


function (obj::PartitionedTensorTrain{T,N})(
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::T where {T,N}
    if !iscompatible(obj, indexsets)
        return zero(T)
    end
    return obj.data(indexsets)
end


function (obj::PartitionedTensorTrain{T,N})(indexset::MultiIndex)::T where {T,N}
    ci = CartesianIndices(ntuple(i -> N, length(obj)))
    indexsets = collect(ci.(indexset))
    return obj(indexsets)
end

#function (obj::PartitionedTensorTrain{T,N})(
    #localset::AbstractVector{<:AbstractVector{Int}},
    #Iset::Vector{MultiIndex},
    #Jset::Vector{MultiIndex})::T where {T,N}
#end


function partition!(obj::PartitionedTensorTrain{T,N}, prj::Vector{Int}, targetlegg::Int)::PartitionedTensorTrain{T,N} where {T,N}
    obj.project[targetlegg] .= prj
    return obj
end


"""
Return if two partitions are compatible.
Entry 0 denotes this dimension is not partitioned.
Positive enttries denote the indices of the partition.
"""
function iscompatible(p1::Vector{Int}, p2::Vector{Int})::Bool
    @show p1, p2
    all(p1 .>= 0) || error("p1 must be non-negative")
    all(p2 .>= 0) || error("p2 must be non-negative")
    length(p1) == length(p2) || error("Length mismatch")

    for (i, j) in zip(p1, p2)
        if i == 0 || j == 0
            continue
        end
        if i != j
            return false
        end
    end
    return true
end

#function iscompatible(
    #p1::AbstractVector{<:AbstractVector{Int}},
    #p2::AbstractVector{<:AbstractVector{Int}})::Bool
    #length(p1) == length(p2) || error("Length mismatch")
    #N = length(p1)
    #for n in 1:N
        #iscompatible(p1[n], p
    #end
#end 