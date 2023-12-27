# General localset is not supported!
mutable struct PartitionedTensorTrain{T,N} <: TCI.BatchEvaluator{T}
    data::TensorTrain{T,N}
    projector::Vector{Vector{Int}} # (L, N)
    sitedims::Vector{Vector{Int}} # (L, N)
end

function PartitionedTensorTrain{T,N}(data::TensorTrain{T,N}) where {T,N} projector = [fill(0, N - 2) for _ in 1:length(data)]
    PartitionedTensorTrain{T,N}(data, projector)
end

function PartitionedTensorTrain{T,N}(data, projector;
    compression::Bool=false,
    cutoff::Float64=1e-30,
    maxdim::Int=typemax(Int)) where {T,N}
    L = length(data)
    length(projector) == L || error("Length mismatch: projector")
    obj = PartitionedTensorTrain{T,N}(data, projector, TCI.sitedims(data))
    partition!(obj, projector; compression=compression, cutoff=cutoff, maxdim=maxdim)
    return obj
end

Base.length(obj::PartitionedTensorTrain{T,N}) where {T,N} = length(obj.data)

function iscompatible(obj::PartitionedTensorTrain{T,N},
    indexsets::AbstractVector{<:AbstractVector{LocalIndex}})::Bool where {T,N}
    for n in 1:N-2
        if !iscompatible(
            [p[n] for p in obj.projector],
            [i[n] for i in indexsets])
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

function _multii(obj::PartitionedTensorTrain{T,N}, indexset::MultiIndex)::Vector{Vector{Int}} where {T,N}
    return [
        collect(Tuple(CartesianIndices(Tuple(obj.sitedims[l]))[i]))
        for (l, i) in enumerate(indexset)]
end

# Evaluate the object at a single linear indexset
function (obj::PartitionedTensorTrain{T,N})(indexset::MultiIndex)::T where {T,N}
    return obj(_multii(obj, indexset))
end

function projectat!(A::Array{T,N}, idxpos, targetidx)::Array{T,N} where {T,N}
    mask = [v != targetidx for v in 1:size(A, idxpos)]
    indices = [d == idxpos ? mask : (:) for d in 1:N]
    A[indices...] .= 0.0
    return A
end

function partition!(
    obj::PartitionedTensorTrain{T,N},
    prj::AbstractVector{<:AbstractVector{Int}};
    compression::Bool=false,
    cutoff::Float64=1e-30,
    maxdim::Int=typemax(Int)
)::PartitionedTensorTrain{T,N} where {T,N}

    # TODO: Introduce check for projector compatibility
    obj.projector = deepcopy(prj)

    # Projection
    for l in 1:length(obj.sitedims)
        for n in 1:N-2
            if obj.projector[l][n] == 0
                continue
            end
            projectat!(obj.data.T[l], n + 1, obj.projector[l][n])
        end
    end

    if compression
        obj.data = truncate(obj.data; cutoff=cutoff, maxdim=maxdim)
    end

    return obj
end


# TODO: Remove ITensor dependency
function truncate(obj::TensorTrain{T,N}; cutoff=1e-30, maxdim=typemax(Int))::TensorTrain{T,N} where {T,N}
    sitedims = TCI.sitedims(obj)
    L = length(sitedims)
    sitedims_li = [prod(sitedims[l]) for l in 1:L]
    sites_li = [Index(sitedims_li[l], "n=$l") for l in 1:L]

    tensors = [copy(reshape(t, size(t, 1), :, size(t, 4))) for t in obj.T]

    linkdims = vcat(1, TCI.linkdims(obj), 1)

    links = [Index(s, "link=$(l-1)") for (l, s) in enumerate(linkdims)]
    itensors = [
        ITensor(t, links[l], sites_li[l], links[l+1]) for (l, t) in enumerate(tensors)
    ]
    itensors[1] *= onehot(links[1] => 1)
    itensors[end] *= onehot(links[end] => 1)

    Ψ = MPS(itensors)
    truncate!(Ψ; maxdim=maxdim, cutoff=cutoff)

    links_new = linkinds(Ψ)
    linkdims_new = [1, dim.(links_new)..., 1]

    tensors_truncated = Array{T,4}[]
    for l in 1:L
        t =
            if l == 1
                Array(Ψ[1], (sites_li[1], links_new[1]))
            elseif l == L
                Array(Ψ[end], (links_new[end], sites_li[end]))
            else
                Array(Ψ[l], (links_new[l-1], sites_li[l], links_new[l]))
            end

        push!(
            tensors_truncated,
            reshape(t,
                linkdims_new[l], sitedims[l]..., linkdims_new[l+1]
            )
        )
    end

    return TensorTrain{T,N}(tensors_truncated)
end


"""
Return if two partitions are compatible.
Entry 0 denotes this dimension is not partitioned.
Positive enttries denote the indices of the partition.
"""
function iscompatible(p1::Vector{Int}, p2::Vector{Int})::Bool
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