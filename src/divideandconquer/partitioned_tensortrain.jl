"""
Collection of ProjectableEvaluator objects

The underlying data will be copied when projected.
"""
mutable struct PartitionedTensorTrain{T} <: ProjectableEvaluator{T}
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

function Base.show(io::IO, obj::PartitionedTensorTrain{T}) where {T}
    return print(
        io, "PartitionedTensorTrain{$T} consisting of $(length(obj.tensortrains)) TTs"
    )
end

#function Base.show(io::IO, obj::PartitionedTensorTrain{T}) where {T}
#print(io, "PartitionedTensorTrain{$T}")
#for tt in obj.tensortrains
#print(io, "  ", tt, " ")
#end
#end

"""
Sum over external indices
"""
function sum(obj::PartitionedTensorTrain{T})::T where {T}
    return Base.sum(sum.(obj.tensortrains))
end

# multi-site-index evaluation
function (obj::PartitionedTensorTrain{T})(indexset::MMultiIndex)::T where {T}
    if !(indexset <= obj.projector)
        return zero(T)
    end
    return Base.sum((t(indexset) for t in obj.tensortrains))
end

# multi-site-index evaluation
function (obj::PartitionedTensorTrain{T})(
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]

    return obj(leftindexset_, rightindexset_, Val(M))
end

# single-site-index evaluation
function (obj::PartitionedTensorTrain{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    println("PartitionedTensorTrain batch")
    t1 = time_ns()
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
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
            if hasoverlap(
                Projector(l_full, obj.tensortrains[ip].projector.sitedims),
                obj.tensortrains[ip].projector,
            )
                push!(left_mask[ip], il)
                push!(leftindexset_[ip], l)
            end
        end
    end

    # Find out which tensortrains are needed for the given right indexsets
    for (ir, r) in enumerate(rightindexset)
        r_full = multii(obj.sitedims, vcat(fill(0, L - length(r)), r))
        for (ip, p) in enumerate(obj.tensortrains)
            #@show r_full
            #@show obj.tensortrains[ip].projector.data
            if hasoverlap(
                Projector(r_full, obj.tensortrains[ip].projector.sitedims),
                obj.tensortrains[ip].projector,
            )
                push!(right_mask[ip], ir)
                push!(rightindexset_[ip], r)
            end
        end
    end
    t2 = time_ns()

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
        @show ip, length(leftindexset_[ip]), length(rightindexset_[ip])
        result_ = obj.tensortrains[ip](leftindexset_[ip], rightindexset_[ip], Val(M))
        result[left_mask[ip], .., right_mask[ip]] .+= result_
    end
    t3 = time_ns()
    println("Time: ", (t2 - t1) / 1e6, " ", (t3 - t2) / 1e6)

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
            prj_new = deepcopy(internal_obj.projector)
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

function Base.reshape(
    obj::PartitionedTensorTrain{T}, dims::AbstractVector{<:AbstractVector{Int}}
)::PartitionedTensorTrain{T} where {T}
    tensortrains = ProjectableEvaluator{T}[reshape(x, dims) for x in obj.tensortrains]
    return PartitionedTensorTrain(tensortrains, reshape(obj.projector, dims), dims)
end

function create_multiplier(
    ptt1::PartitionedTensorTrain{T}, ptt2::PartitionedTensorTrain{T}
)::PartitionedTensorTrain{T} where {T}
    globalprojector = Projector(
        [[x[1], y[2]] for (x, y) in zip(ptt1.projector, ptt2.projector)],
        [[x[1], y[2]] for (x, y) in zip(ptt1.sitedims, ptt2.sitedims)],
    )
    return create_multiplier(
        Vector{ProjectedTensorTrain{T,4}}(ptt1.tensortrains),
        Vector{ProjectedTensorTrain{T,4}}(ptt2.tensortrains),
        globalprojector,
    )
end
