"""
TensorTrain projected on a subset of indices

The underlying data will be copied when projected.
"""
mutable struct ProjectedTensorTrain{T,N} <: ProjectableEvaluator{T}
    data::TensorTrain{T,N}
    cache::TCI.TTCache{T}
    projector::Projector # (L, N-2)
    sitedims::Vector{Vector{Int}} # (L, N-2)

    function ProjectedTensorTrain{T,N}(
        tt::TensorTrain{T,N}, projector::Projector, sitedims::Vector{Vector{Int}}
    ) where {T,N}
        return new{T,N}(
            tt,
            TCI.TTCache(TensorTrain{T,3}(tt, collect(prod.(sitedims)))),
            projector,
            sitedims)
    end
end

function ProjectedTensorTrain(
    tt::TensorTrain{T,N}, projector::Projector, sitedims::Vector{Vector{Int}}
) where {T,N}
    return ProjectedTensorTrain{T,N}(tt, projector, sitedims)
end

function ProjectedTensorTrain(tt::TensorTrain{T,N}, projector; kwargs...) where {T,N}
    return ProjectedTensorTrain{T,N}(tt, projector; kwargs...)
end

function ProjectedTensorTrain(tt::TensorTrain{T,N}) where {T,N}
    return ProjectedTensorTrain(
        tt,
        Projector([fill(0, N - 2) for _ in 1:length(tt)], TCI.sitedims(tt)),
        TCI.sitedims(tt),
    )
end

function ProjectedTensorTrain{T,N}(
    tt, projector; compression::Bool=false, cutoff::Float64=1e-30, maxdim::Int=typemax(Int)
) where {T,N}
    L = length(tt)
    length(projector) == L || error("Length mismatch: projector")
    obj = ProjectedTensorTrain{T,N}(tt, projector, TCI.sitedims(tt))
    # Why do we need force option?
    globalprojector = Projector([fill(0, N - 2) for _ in 1:length(tt)], projector.sitedims)
    obj = project(
        obj,
        projector;
        force=(project != globalprojector),
        compression=compression,
        cutoff=cutoff,
        maxdim=maxdim,
    )
    return obj
end

# This function is type unstable
function Base.reshape(
    obj::ProjectedTensorTrain{T,N}, dims::AbstractVector{<:AbstractVector{Int}}
) where {T,N}
    length(unique(length.(dims))) == 1 ||
        error("The number of siteindices must be the same at all tensors!")
    N2 = Base.only(unique(length.(dims))) + 2

    ttdata = [
        reshape(obj.data[n], size(obj.data[n])[1], dims[n]..., size(obj.data[n])[end]) for
        n in eachindex(dims)
    ]

    return ProjectedTensorTrain{T,N2}(
        TensorTrain{T,N2}(ttdata), reshape(obj.projector, dims), dims
    )
end

Base.length(obj::ProjectedTensorTrain{T,N}) where {T,N} = length(obj.data)

function Base.show(io::IO, obj::ProjectedTensorTrain{T,N}) where {T,N}
    return print(
        io,
        "ProjectedTensorTrain{$T} with rank $(TCI.rank(obj.data)) on $(obj.projector.data)",
    )
end

function TCI.sitetensors(obj::ProjectedTensorTrain{T,N}) where {T,N}
    return obj.data.sitetensors
end

function TCI.sitetensor(obj::ProjectedTensorTrain{T,N}, i) where {T,N}
    return TCI.sitetensor(obj.tt, i)
end

function batchevaluateprj(
    obj::ProjectedTensorTrain{T,N},
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,N,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    L = length(obj)
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]
    projector = Int[
        isprojectedat(obj.projector, n) ? _lineari(obj.sitedims[n], obj.projector[n]) : 0
        for n in (NL + 1):(L - NR)
    ]
    returnshape = [
        isprojectedat(obj.projector, n) ? 1 : prod(obj.sitedims[n])
        for n in (NL + 1):(L - NR)
    ]
    res = TCI.batchevaluate(obj.cache, leftindexset_, rightindexset_, Val(M), projector)
    return reshape(res, length(leftindexset), returnshape..., length(rightindexset))
end

function projectat!(A::Array{T,N}, idxpos, targetidx)::Array{T,N} where {T,N}
    mask = [v != targetidx for v in 1:size(A, idxpos)]
    indices = [d == idxpos ? mask : (:) for d in 1:N]
    A[indices...] .= 0.0
    return A
end

function project(
    obj::ProjectedTensorTrain{T,N},
    prj::Projector;
    force::Bool=false,
    compression::Bool=false,
    cutoff::Float64=1e-30,
    maxdim::Int=typemax(Int),
)::Union{ProjectedTensorTrain{T,N},Nothing} where {T,N}
    prj <= obj.projector || error("Projector $prj is not compatible with $obj.projector")

    if prj == obj.projector && !force
        return obj
    end

    # Make a copy
    projector = deepcopy(prj)
    data = deepcopy(obj.data)

    # Projection
    for l in 1:length(obj.sitedims)
        for n in 1:(N - 2)
            if projector[l][n] == 0
                continue
            end
            if (force && projector[l][n] > 0) || projector[l][n] != obj.projector[l][n]
                projectat!(data[l], n + 1, projector[l][n])
            end
        end
    end

    if compression
        data = truncate(data; cutoff=cutoff, maxdim=maxdim)
    end

    return ProjectedTensorTrain{T,N}(data, projector, obj.sitedims)
end

# TODO: Remove ITensor dependency
function truncate(
    obj::TensorTrain{T,N}; cutoff=1e-30, maxdim=typemax(Int)
)::TensorTrain{T,N} where {T,N}
    sitedims = TCI.sitedims(obj)
    L = length(sitedims)
    sitedims_li = [prod(sitedims[l]) for l in 1:L]
    sites_li = [Index(sitedims_li[l], "n=$l") for l in 1:L]

    tensors = [copy(reshape(t, size(t, 1), :, size(t, 4))) for t in obj]

    linkdims = vcat(1, TCI.linkdims(obj), 1)

    links = [Index(s, "link=$(l-1)") for (l, s) in enumerate(linkdims)]
    itensors = [
        ITensor(t, links[l], sites_li[l], links[l + 1]) for (l, t) in enumerate(tensors)
    ]
    itensors[1] *= onehot(links[1] => 1)
    itensors[end] *= onehot(links[end] => 1)

    Ψ = MPS(itensors)
    truncate!(Ψ; maxdim=maxdim, cutoff=cutoff)

    links_new = linkinds(Ψ)
    linkdims_new = [1, dim.(links_new)..., 1]

    tensors_truncated = Array{T,4}[]
    for l in 1:L
        t = if l == 1
            Array(Ψ[1], (sites_li[1], links_new[1]))
        elseif l == L
            Array(Ψ[end], (links_new[end], sites_li[end]))
        else
            Array(Ψ[l], (links_new[l - 1], sites_li[l], links_new[l]))
        end

        push!(
            tensors_truncated,
            reshape(t, linkdims_new[l], sitedims[l]..., linkdims_new[l + 1]),
        )
    end

    return TensorTrain{T,N}(tensors_truncated)
end