"""
Collection of ProjectableEvaluator objects
"""
mutable struct PartitionedTensorTrain{T}
    tensortrains::Vector{ProjectableEvaluator{T}}
    # This PartitionedTensorTrain is projected on the indices specified by this projector
    projector::Projector
    sitedims::Vector{Vector{Int}}

    # Cache
    #cache_leftindexset::Dict{MultiIndex,Vector{Int}}
    #cache_rightindexset::Dict{MultiIndex,Vector{Int}}

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
    L = length(obj.products[1].sitedims)

    left_mask = [Int[] for _ in 1:obj.products]
    right_mask = [Int[] for _ in 1:obj.products]
    leftindexset_ = [MultiIndex[] for _ in 1:obj.products]
    rightindexset_ = [MultiIndex[] for _ in 1:obj.products]

    # Find out which products are needed for the given left indexsets
    for l in leftindexset
        l_full = vcat(l, fill(0, L - length(l)))
        for (ip, p) in enumerate(1:obj.products)
            if l_full <= obj.tensortrains[p].projector
                push!(left_mask[p], ip)
                push!(leftindexset_[p], l_full)
            end
        end
    end

    # Find out which products are needed for the given right indexsets
    for r in rightindexset
        r_full = vcat(fill(0, L - length(r)), r)
        for (ip, p) in enumerate(1:obj.products)
            if r_full <= obj.tensortrains[p].projector
                push!(right_mask[p], ip)
                push!(rightindexset_[p], r_full)
            end
        end
    end

    result = zeros(T, length(leftindexset), prod.(obj.sitedims[nl+1:nl+M])..., length(rightindexset))
    for ip in 1:obj.products
        if length(leftindexset_[ip]) * length(rightindexset_[ip]) == 0
            continue
        end
        result_ = obj.products[ip](leftindexset_[ip], rightindexset_[ip], Val(M))
        result[left_mask[ip], .., right_mask[ip]] .+= result_
    end

    return result
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
    #empty!(obj.cache_leftindexset)
    #empty!(obj.cache_rightindexset)
    return obj
end
