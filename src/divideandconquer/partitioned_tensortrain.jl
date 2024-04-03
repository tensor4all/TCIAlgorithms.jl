"""
Collection of ProjectableEvaluator objects

The underlying data will be copied when projected.
"""
mutable struct PartitionedTensorTrain{T}
    tensortrains::Vector{ProjectableEvaluator{T}}
    # This PartitionedTensorTrain is projected on
    # the indices specified by `projector`.
    # All items in `tensortrains` must be compatible with `projector`.
    projector::Projector
    sitedims::Vector{Vector{Int}}

    function PartitionedTensorTrain(
        tensortrains::AbstractVector{<:ProjectableEvaluator{T}}, projector, sitedims
    ) where {T}
        for t in tensortrains
            t.projector <= projector || error("Projector mismatch")
        end
        return new{T}(tensortrains, projector, sitedims)
    end

    function PartitionedTensorTrain(internalobj::ProjectableEvaluator{T}) where {T}
        return new{T}([internalobj], internalobj.projector, internalobj.sitedims)
    end
end

"""
Sum over external indices
"""
function sum(obj::PartitionedTensorTrain{T})::T where {T}
    return Base.sum(sum.(obj.tensortrains))
end

function (obj::PartitionedTensorTrain{T})(
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}}
)::T where {T}
    if !(indexsets <= obj.projector)
        return zero(T)
    end
    return Base.sum((t(indexsets) for t in obj.tensortrains))
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
    result = zeros(
        T,
        length(leftindexset),
        prod.(obj.sitedims[(nl + 1):(nl + M)])...,
        length(rightindexset),
    )
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
    obj::PartitionedTensorTrain{T}, prj::Projector
)::PartitionedTensorTrain{T} where {T}
    prj <= projector(obj) || error("Projector mismatch")
    for (i, t) in enumerate(obj.tensortrains)
        obj.tensortrains[i] = prj
    end
    obj.projector = prj
    return obj
end

function partitionat(
    obj::PartitionedTensorTrain{T},
    siteidx::Int;
    compression::Bool=false,
    cutoff::Float64=1e-30,
    maxdim::Int=typemax(Int),
)::PartitionedTensorTrain{T} where {T}
    tts = ProjectableEvaluator{T}[]

    new_indices = collect(
        typesafe_iterators_product(
            Val(length(obj.sitedims[siteidx])), obj.sitedims[siteidx]
        ),
    )
    for internal_obj in obj.tensortrains
        all(internal_obj.projector[siteidx] .== 0) ||
            error("Some of site indices at $siteidx are already projected")

        for (i, new_idx) in enumerate(new_indices)
            prj_new = copy(internal_obj.projector)
            prj_new.data[siteidx] .= new_idx
            push!(
                tts,
                project(
                    internal_obj,
                    prj_new;
                    compression=compression,
                    cutoff=cutoff,
                    maxdim=maxdim,
                ),
            )
        end
    end

    return PartitionedTensorTrain(tts, obj.projector, obj.sitedims)
end
