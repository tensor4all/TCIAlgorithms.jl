"""
Represents a TensorTrain object which can be projected on a subset of indices

Compared to TCI.TensorTrain, this object has additional functionalities:
* Projection
* Multi site indices per tensor are supported.
* Fast evaluation by caching intermediate objects of contraction
"""
mutable struct ProjTensorTrain{T} <: ProjectableEvaluator{T}
    data::TensorTrain{T}
    cache::TCI.TTCache{T}
    projector::Projector
    sitedims::Vector{Vector{Int}}

    function ProjTensorTrain{T}(tt::TensorTrain{T,N}, projector::Projector) where {T,N}
        sitedims = projector.sitedims
        for (n, tensor) in enumerate(tt)
            prod(size(tensor)[2:(end - 1)]) == prod(sitedims[n]) ||
                error("The site indices must match the tensor size")
        end
        tts = TensorTrain{T,3}(tt, collect(prod.(sitedims)))

        # check if the projector is compatible with the data
        if !_check_projector_compatibility(projector, tts)
            error("The projector is not compatible with the data")
        end

        return new{T}(
            tts, TCI.TTCache(tts, projector.sitedims), projector, projector.sitedims
        )
    end

    function ProjTensorTrain{T}(tt::TensorTrain{T,N}) where {T,N}
        tts = TensorTrain{T,3}(tt, [[prod(s)] for s in TCI.sitedims(tt)])
        projector = Projector([fill(0, N - 2) for _ in 1:length(tt)], TCI.sitedims(tt))
        if !_check_projector_compatibility(projector, tts)
            error("The projector is not compatible with the data")
        end
        return ProjTensorTrain{T}(tts, projector)
    end
end

function _check_projector_compatibility(
    projector::Projector, tts::TensorTrain{T,3}
) where {T}
    sitedims = projector.sitedims
    for n in 1:length(tts)
        if all(projector[n] .== 0)
            continue
        end
        mask = (
            projector[n][s] == 0 ? Colon() : Not(projector[n][s], sitedims[n][s]) for
            s in 1:length(sitedims[n])
        )
        sitetensor = reshape(tts[n], size(tts[n])[1], sitedims[n]..., size(tts[n])[end])
        if !(all(sitetensor[:, mask..., :] .== 0.0))
            return false
        end
    end
    return true
end

"""
The user must make sure that the data is compatible with the given projector.
We do not recommend to use this function directly.
Use `project(ProjTensorTrain(tt), projector)` instead.
"""
function ProjTensorTrain(tt::TensorTrain{T,N}, projector::Projector) where {T,N}
    return ProjTensorTrain{T}(tt, projector)
end

function ProjTensorTrain(tt::TensorTrain{T,N}) where {T,N}
    return ProjTensorTrain{T}(tt)
end

function (obj::ProjTensorTrain{T})(indexset::MMultiIndex)::T where {T}
    return obj.data(lineari(obj.sitedims, indexset))
end

function Base.reshape(
    obj::ProjTensorTrain{T}, sitedims::AbstractVector{<:AbstractVector{Int}}
)::ProjTensorTrain{T} where {T}
    length(unique(length.(sitedims))) == 1 ||
        error("The number of siteindices must be the same at all tensors!")
    prod.(sitedims) == prod.(obj.sitedims) ||
        error("The total number of siteindices must be the same at all tensors!")

    return ProjTensorTrain{T}(obj.data, reshape(obj.projector, sitedims), sitedims)
end

Base.length(obj::ProjTensorTrain{T}) where {T} = length(obj.data)

function Base.show(io::IO, obj::ProjTensorTrain{T}) where {T}
    return print(io, "ProjTensorTrain{$T} projected on $(obj.projector.data)")
end

function TCI.sitetensors(obj::ProjTensorTrain{T}) where {T}
    return [TCI.sitetensor(obj, i) for i in 1:length(obj)]
end

function TCI.sitetensor(obj::ProjTensorTrain{T}, i) where {T}
    tensor = TCI.sitetensor(obj.data, i)
    return reshape(tensor, size(tensor, 1), obj.sitedims[i]..., size(tensor)[end])
end

function batchevaluateprj(
    obj::ProjTensorTrain{T},
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])
    L = length(obj)
    leftmmultiidxset_ = [lineari(obj.sitedims[1:NL], x) for x in leftmmultiidxset]
    rightmmultiidxset_ = [
        lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset
    ]
    projector = [obj.projector[n] for n in (NL + 1):(L - NR)]
    returnshape = projectedshape(obj.projector, NL + 1, L - NR)
    res = TCI.batchevaluate(
        obj.cache, leftmmultiidxset_, rightmmultiidxset_, Val(M), projector
    )
    return reshape(res, length(leftmmultiidxset), returnshape..., length(rightmmultiidxset))
end

function projectat!(A::Array{T,N}, idxpos, targetidx)::Array{T,N} where {T,N}
    mask = [v != targetidx for v in 1:size(A, idxpos)]
    indices = [d == idxpos ? mask : (:) for d in 1:N]
    A[indices...] .= 0.0
    return A
end

function project(
    obj::ProjTensorTrain{T},
    prj::Projector;
    compression::Bool=false,
    tolerance::Float64=1e-12,
    maxbonddim::Int=typemax(Int),
)::ProjTensorTrain{T} where {T}
    prj <= obj.projector || error("Projector $prj is not compatible with $obj.projector")
    prj.sitedims == obj.sitedims ||
        error("sitedims mismatch $(prj.sitedims) and $(obj.sitedims)")

    if obj.projector <= prj && _check_projector_compatibility(prj, obj.data)
        # No need to project
        return obj
    end

    # Make a copy
    projector = deepcopy(prj)
    data = deepcopy(obj.data) # each tensor core is a four-dimensional array

    # Projection
    _project_tt!(data, projector)
    if compression
        data = truncate(data; tolerance=tolerance, maxbonddim=maxbonddim)
        _project_tt!(data, projector)
    end

    return ProjTensorTrain{T}(data, projector)
end

function _project_tt!(tt::TensorTrain{T,3}, projector::Projector) where {T}
    sitedims = projector.sitedims

    for l in 1:length(sitedims)
        if all(projector[l] .== 0)
            # No projection
            continue
        end
        sitetensor_multii = reshape(tt[l], size(tt[l])[1], sitedims[l]..., size(tt[l])[end])
        for s in 1:length(projector[l])
            if projector[l][s] == 0
                continue
            end
            projectat!(sitetensor_multii, s + 1, projector[l][s])
        end
    end
end

function truncate(
    obj::TensorTrain{T,N}; tolerance=1e-14, maxbonddim=typemax(Int)
)::TensorTrain{T,N} where {T,N}
    tt = deepcopy(obj)
    TCI.compress!(tt, :SVD; tolerance=tolerance, maxbonddim=maxbonddim)
    return tt
end

function makeprojectable(tt::TensorTrain{T,N}) where {T,N}
    return ProjTensorTrain(tt)
end
