mutable struct PartitionedTensorTrain{T}
    tensortrains::Vector{ProjectableEvaluator{T}}
    # This PartitionedTensorTrain is projected on the indices specified by this projector
    projector::Projector
    sitedims::Vector{Vector{Int}}

    function PartitionedTensorTrain{T}(tensortrains, projector, sitedims) where {T}
        for t in tensortrains
            t.projector <= projector || error("Projector mismatch")
        end
        return new{T}(tensortrains, projector, sitedims)
    end

end


#PartitionedTensorTrain(tensortrains::ProjectedTensorTrain{T,N}, projector, sitedims) where {T,N} = PartitionedTensorTrain{T,N}(tensortrains::ProjectedTensorTrain{T,N}, projector, sitedims)


function (obj::PartitionedTensorTrain{T})(
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::T where {T}
    if !(indexsets <= obj.projector)
        return zero(T)
    end
    return sum((t(indexsets) for t in obj.tensortrains))
end


function (obj::PartitionedTensorTrain{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,N,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return zeros(T, 0, 0)
    end
    # TODO: Optimize
    return sum((v(indexset) for v in obj.products))
end


function project(
    obj::PartitionedTensorTrain{T},
    prj::Projector
)::PartitionedTensorTrain{T} where {T}
    prj <= projector(obj) || error("Projector mismatch")
    for (i, t) in enumerate(obj.tensortrains)
        obj.tensortrains[i] = prj
    end
    obj.projector = prj
    return obj
end
