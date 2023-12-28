#==
mutable struct PartitionedTensorTrain{T,N} <: ProjectableEvaluator{T}
    tensortrains::OrderedSet{ProjectedTensorTrain{T,N}}
    # This PartitionedTensorTrain is projected on the indices specified by this projector
    projector::Projector
    sitedims::Vector{Vector{Int}}
end


PartitionedTensorTrain(tensortrains::ProjectedTensorTrain{T,N}, projector, sitedims) where {T,N} = PartitionedTensorTrain{T,N}(tensortrains::ProjectedTensorTrain{T,N}, projector, sitedims)

function PartitionedTensorTrain{T,N}(tensortrains::ProjectedTensorTrain{T,N}, projector, sitedims) where {T,N}
    for t in tensortrains
        projector(t) <= projector || error("Projector mismatch")
    end
    return PartitionedTensorTrain(tensortrains, projector, sitedims)
end

function (obj::PartitionedTensorTrain{T,N})(
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::T where {T,N}
    if projector(obj) <= indexsets
        return zero(T)
    end
    return sum((t(indexsets) for t in obj.tensortrains))
end

function project!(
    obj::PartitionedTensorTrain{T,N},
    prj::Projector
)::PartitionedTensorTrain{T,N} where {T,N}
    prj <= projector(obj) || error("Projector mismatch")
    for t in obj.tensortrains
        project!(t, prj)
    end
    obj.projector = prj
    obj
end
==#