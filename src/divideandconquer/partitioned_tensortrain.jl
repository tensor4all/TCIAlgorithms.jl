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
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return zeros(T, 0, 0)
    end
    L = length(obj.tensortrains[1].sitedims)

    left_mask = [Int[] for _ in obj.tensortrains]
    right_mask = [Int[] for _ in obj.tensortrains]
    leftindexset_ = [MultiIndex[] for _ in obj.tensortrains]
    rightindexset_ = [MultiIndex[] for _ in obj.tensortrains]

    # Find out which tensortrains are needed for the given left indexsets
    for (il, l) in enumerate(leftindexset)
        l_full = multii(obj.sitedims, vcat(l, fill(0, L - length(l))))
        for (ip, p) in enumerate(obj.tensortrains)
            if hasoverlap(Projector(l_full), obj.tensortrains[ip].projector)
                push!(left_mask[ip], il)
                push!(leftindexset_[ip], l)
            end
        end
    end

    # Find out which tensortrains are needed for the given right indexsets
    for (ir, r) in enumerate(rightindexset)
        r_full = multii(obj.sitedims, vcat(fill(0, L - length(r)), r))
        for (ip, p) in enumerate(obj.tensortrains)
            if hasoverlap(Projector(r_full), obj.tensortrains[ip].projector)
                push!(right_mask[ip], ir)
                push!(rightindexset_[ip], r)
            end
        end
    end

    nl = length(first(leftindexset))
    result = zeros(T, length(leftindexset), prod.(obj.sitedims[nl+1:nl+M])..., length(rightindexset))
    for ip in 1:length(obj.tensortrains)
        if length(leftindexset_[ip]) * length(rightindexset_[ip]) == 0
            continue
        end
        result_ = obj.tensortrains[ip](leftindexset_[ip], rightindexset_[ip], Val(M))
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
