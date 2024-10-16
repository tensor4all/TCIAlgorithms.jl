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

#function linkdims(obj::ProjTensorTrain)
#TCI.linkdims(obj.data)
#end

function _check_projector_compatibility(
    projector::Projector, tts::TensorTrain{T,3}
) where {T}
    sitedims = projector.sitedims
    return reduce(
        &,
        (
            _check_projector_compatibility(projector[n], sitedims[n], tts[n]) for
            n in 1:length(tts)
        ),
    )
end

function _check_projector_compatibility(
    projector::Vector{Int}, sitedims::Vector{Int}, tensor::AbstractArray{T,3}
) where {T}
    if all(projector .== 0)
        return true
    end
    mask = (
        projector[s] == 0 ? Colon() : Not(projector[s], sitedims[s]) for
        s in 1:length(sitedims)
    )
    sitetensor = reshape(tensor, size(tensor)[1], sitedims..., size(tensor)[end])
    if all(sitetensor[:, mask..., :] .== 0.0)
        return true
    else
        badinds = findall(!iszero, sitetensor[:, mask..., :])
        @show badinds sitetensor[:, mask..., :][badinds]
        return false
    end
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

function ProjTensorTrain(
    tt::TensorTrain{T,N}, sitedims::AbstractVector{<:AbstractVector{<:Integer}}
) where {T,N}
    return reshape(ProjTensorTrain{T}(tt), sitedims)
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

    return ProjTensorTrain{T}(obj.data, reshape(obj.projector, sitedims))
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
    prj.sitedims == obj.sitedims ||
        error("sitedims mismatch $(prj.sitedims) and $(obj.sitedims)")

    if obj.projector <= prj && _check_projector_compatibility(prj, obj.data)
        #@show "No projection", objectid(obj.projector.data)
        # No need to project
        return ProjTensorTrain{T}(obj.data, deepcopy(obj.projector))
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
    #@show objectid(obj.projector.data), objectid(r.projector.data)
    #return r
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

function approxtt(
    obj::ProjTensorTrain{T}; maxbonddim=typemax(Int), tolerance=1e-14, kwargs...
)::ProjTensorTrain{T} where {T}
    return project(
        ProjTensorTrain(
            truncate(obj.data; tolerance=tolerance, maxbonddim=maxbonddim), obj.sitedims
        ),
        obj.projector,
    )
end

function isapproxttavailable(obj::ProjTensorTrain)
    return true
end

function add(
    a::ProjTensorTrain{T},
    b::ProjTensorTrain{T};
    maxbonddim=typemax(Int),
    tolerance=1e-14,
    kwargs...,
)::ProjTensorTrain{T} where {T}
    # HS: TCI.add does not use a relative tolerance.
    # For the moment, we need to use ITensors.add instead
    a_MPS = MPS(a.data)
    b_MPS = MPS(b.data; sites=siteinds(a_MPS))
    ab_MPS = +(a_MPS, b_MPS; alg="directsum")
    truncate!(ab_MPS; maxdim=maxbonddim, cutoff=tolerance^2)
    ab = reshape(ProjTensorTrain(TensorTrain{T,3}(MPO([x for x in ab_MPS]))), a.sitedims)
    return project(ab, a.projector | b.projector)
end

"""
Project the tensor train on the subset of site indices.
The returned tensor train object has the number of site indices equal to the number of the unprojected subset in the input.
The dimension of the unprojected site indices are the same as the original ones, i.e., before the projection.
On each tensor core in the input tensor train, site indices are either projected or not projected.
The data will be copied, and in the returned tensor train, the site indices are fused.
"""
function project_on_subsetsiteinds(obj::ProjTensorTrain{T}) where {T}
    tensors = Vector{Array{T,3}}(undef, length(obj.data))
    projected = fill(false, length(obj.data))
    for i in 1:length(obj.data)
        if all(obj.projector[i] .== 0)
            tensors[i] = _to_3d_array(obj.data[i])
            continue
        elseif all(obj.projector[i] .> 0)
            tensors[i] = _to_3d_array(
                _from_3d_array(obj.data[i], obj.sitedims[i])[:, obj.projector[i]..., :]
            )
            projected[i] = true
        else
            error("At site $i, all site indices are not projected or projected")
        end
        tensors[i] = _to_3d_array(tensors[i])
    end

    tensor_merged = deepcopy(tensors)
    to_be_merged::Vector{Bool} = deepcopy(projected)
    while count(to_be_merged) > 0
        tensor_merged, to_be_merged = _merge_projected(tensor_merged, to_be_merged)
    end

    !any(to_be_merged) == true || error("Merging of projected tensors failed")

    return TensorTrain{T,3}(tensor_merged)
end

function _merge_projected(tensor_merged::Vector{Array{T,3}}, to_be_merged) where {T}
    tensor_merged_ = Array{T,3}[]
    to_be_merged_ = Bool[]
    while length(tensor_merged) > 0
        @assert length(tensor_merged) == length(to_be_merged)
        if length(tensor_merged) == 1
            push!(tensor_merged_, popfirst!(tensor_merged))
            push!(to_be_merged_, popfirst!(to_be_merged))
        elseif !to_be_merged[1] && !to_be_merged[2]
            push!(tensor_merged_, popfirst!(tensor_merged))
            push!(to_be_merged_, popfirst!(to_be_merged))
        elseif to_be_merged[1] || to_be_merged[2]
            push!(tensor_merged_, _merge(tensor_merged[1], tensor_merged[2]))
            push!(to_be_merged_, to_be_merged[1] && to_be_merged[2])
            for _ in 1:2
                popfirst!(tensor_merged)
                popfirst!(to_be_merged)
            end
        else
            error("This should not happen")
        end
    end
    return tensor_merged_, to_be_merged_
end

function _merge(A::Array{T,3}, B::Array{T,3}) where {T}
    AB = _contract(A, B, (3,), (1,))
    return _to_3d_array(AB)
end

function _to_3d_array(obj::Array{T,N})::Array{T,3} where {T,N}
    return reshape(obj, size(obj)[1], :, size(obj)[end])
end

function _from_3d_array(obj::Array{T,3}, sitedims) where {T}
    return reshape(obj, size(obj)[1], sitedims..., size(obj)[end])
end

"""
Evaluate `obj` at all possible indexsets and return a full tensor
if `reducesitedims` is true, in the returned tensor, the dimensions of the projected site indices are reduced to 1.
"""
function fulltensor(
    obj::ProjTensorTrain{T}; fused::Bool=false, reducesitedims=false
)::Array{T} where {T}
    sitetensors = Array{T,3}[]
    sitedims = Vector{Int}[]
    for i in 1:length(obj.sitedims)
        sitetensor = reshape(
            obj.data[i], size(obj.data[i])[1], obj.sitedims[i]..., size(obj.data[i])[3]
        )
        if reducesitedims
            p = map(x -> x == 0 ? Colon() : x, obj.projector[i])
            sitetensor_ = sitetensor[:, p..., :]
            push!(sitetensors, _to_3d_array(sitetensor_))
            push!(sitedims, collect(size(sitetensor_)[2:(end - 1)]))
        else
            sitetensor_ = sitetensor
            push!(sitetensors, _to_3d_array(sitetensor_))
            push!(sitedims, collect(size(sitetensor_)[2:(end - 1)]))
        end
    end

    result::Array{T,3} = _merge(sitetensors[1], sitetensors[2])
    for i in 3:length(sitetensors)
        result = _merge(result, sitetensors[i])
    end

    if fused
        returnsize = collect(prod.(sitedims))
    else
        returnsize = collect(Iterators.flatten(sitedims))
    end
    return reshape(result, returnsize...)
end

function zeroprojtt(::Type{T}, projector::Projector)::ProjTensorTrain{T} where {T}
    tt = TensorTrain([zeros(T, 1, d, 1) for d in prod.(projector.sitedims)])
    ptt = ProjTensorTrain(tt, projector.sitedims)
    return project(ptt, projector)
end
