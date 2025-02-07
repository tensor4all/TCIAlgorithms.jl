"""
Specify the ordering of patching
"""
struct PatchOrdering
    ordering::Vector{Int}
    function PatchOrdering(ordering::Vector{Int})
        sort(ordering) == collect(1:length(ordering)) || error("Inconsistent ordering")
        return new(ordering)
    end
end

Base.length(po::PatchOrdering) = length(po.ordering)

function Base.iterate(p::PatchOrdering, state=1)
    if state > length(p.odering)
        return nothing
    end
    return (p.odering[state], state + 1)
end

Base.getindex(p::PatchOrdering, index::Int) = p.ordering[index]

"""
n is the length of the prefix.
"""
function maskactiveindices(po::PatchOrdering, nprefix::Int)
    mask = ones(Bool, length(po.ordering))
    mask[po.ordering[1:nprefix]] .= false
    return mask
end

"""
T: Float64, ComplexF64, etc.
M: TensorCI2, MPS, etc.
"""
abstract type AbstractPatchCreator{T,M} end

mutable struct PatchCreatorResult{T,M}
    data::Union{M,Nothing}
    isconverged::Bool
    resultpivots::Vector{MultiIndex}

    function PatchCreatorResult{T,M}(
        data::Union{M,Nothing}, isconverged::Bool, resultpivots::Vector{MultiIndex}
    )::PatchCreatorResult{T,M} where {T,M}
        return new{T,M}(data, isconverged, resultpivots)
    end

    function PatchCreatorResult{T,M}(
        data::Union{M,Nothing}, isconverged::Bool
    )::PatchCreatorResult{T,M} where {T,M}
        return new{T,M}(data, isconverged, MultiIndex[])
    end
end

function _reconst_prefix(projector::Projector, pordering::PatchOrdering)
    np = Base.sum((isprojectedat(projector, n) for n in 1:length(projector)))
    return [Base.only(projector[n]) for n in pordering.ordering[1:np]]
end

function __taskfunc(creator::AbstractPatchCreator{T,M}, pordering; verbosity=0) where {T,M}
    patch = createpatch(creator)
    prefix::Vector{Int} = _reconst_prefix(creator.projector, pordering)

    if patch.isconverged
        projector = makeproj(pordering, prefix, creator.localdims)
        tt = if patch.data === nothing
            _zerott(T, prefix, pordering, creator.localdims)
        else
            patch.data
        end
        ptt = ProjTensorTrain(tt, projector.sitedims, projector)
        return ptt, nothing
    else
        newtasks = Set{AbstractPatchCreator{T,M}}()
        for ic in 1:creator.localdims[pordering.ordering[length(prefix) + 1]]
            prefix_ = vcat(prefix, ic)
            projector_ = makeproj(pordering, prefix_, creator.localdims)

            # Pivots are shorter, pordering index is in a different position
            active_dims_ = findall(x -> x == [0], creator.projector.data)
            pos_ = findfirst(x -> x == pordering.ordering[length(prefix) + 1], active_dims_)
            pivots_ = [
                copy(piv) for piv in filter(piv -> piv[pos_] == ic, patch.resultpivots)
            ]

            if !isempty(pivots_)
                deleteat!.(pivots_, pos_)
            end

            push!(newtasks, project(creator, projector_; pivots=pivots_))
        end
        return nothing, newtasks
    end
end

function _zerott(T, prefix, po::PatchOrdering, localdims::Vector{Int})
    localdims_ = localdims[maskactiveindices(po, length(prefix))]
    return TensorTrain([zeros(T, 1, d, 1) for d in localdims_])
end

function project(
    obj::AbstractPatchCreator{T,M},
    projector::Projector;
    pivots::Vector{MultiIndex}=MultiIndex[],
) where {T,M}
    projector <= obj.projector || error(
        "Projector $projector is not a subset of the original projector $(obj.f.projector)",
    )

    obj_copy = TCI2PatchCreator{T}(obj) # shallow copy
    obj_copy.projector = deepcopy(projector)
    obj_copy.f = project(obj_copy.f, projector)
    obj_copy.initialpivots = deepcopy(pivots)
    return obj_copy
end

function makeproj(po::PatchOrdering, prefix::Vector{Int}, localdims::Vector{Int})
    data = [[0] for _ in localdims]
    for (i, n) in enumerate(po.ordering[1:length(prefix)])
        data[n][1] = prefix[i]
    end
    return Projector(data, [[x] for x in localdims])
end

function makeproj(
    po::PatchOrdering, prefix::Vector{Vector{Int}}, sitedims::Vector{Vector{Int}}
)
    data = Vector{Int}[fill(0, length(s)) for s in sitedims]
    for (i, n) in enumerate(po.ordering[1:length(prefix)])
        data[n] = deepcopy(prefix[i])
    end
    return Projector(data, sitedims)
end

function makechildproj(proj::Projector, po::PatchOrdering)::Vector{Projector}
    path = createpath(proj, po)
    result = Projector[]

    if length(path) == length(proj)
        return result
    end

    nextprojsite = po.ordering[length(path) + 1]
    for (li, mi) in enumerate(CartesianIndices(Tuple(proj.sitedims[nextprojsite])))
        proj_ = deepcopy(proj)
        proj_[nextprojsite] .= collect(Tuple(mi))
        push!(result, proj_)
    end

    return result
end

"""
`tt` is a TensorTrain{T,3} and `prj` is a Projector.
`tt` is defined on unprojected indices.
Return a ProjTensorTrain{T} defined on full indices.
"""
function ProjTensorTrain(
    tt::TensorTrain{T,3}, localdims::AbstractVector{<:AbstractVector{Int}}, prj::Projector
)::ProjTensorTrain{T} where {T}
    length(tt) == Base.sum((Base.only(p) == 0 for p in prj)) || error("Inconsistent length")
    L = length(prj)

    sitetensors = Array{T,3}[zeros(T, 1, 1, 1) for _ in 1:L]
    linkdims = ones(Int, L + 1)
    localdims_ = [Base.only(d) for d in localdims]

    l_ = 1
    for (l, p) in enumerate(prj)
        onlyp = Base.only(p)
        if onlyp == 0
            linkdims[l] = size(tt[l_], 1)
            linkdims[l + 1] = size(tt[l_], 4)
            sitetensors[l] = tt[l_]
            l_ += 1
        end
    end

    # Compute linkdims on full indices
    while true
        linkdims_ = deepcopy(linkdims)
        for (n, p) in enumerate(prj)
            onlyp = Base.only(p)
            if onlyp != 0
                if linkdims[n] != linkdims[n + 1]
                    linkdims[n] = linkdims[n + 1] = max(linkdims[n], linkdims[n + 1])
                end
            end
        end
        if linkdims == linkdims_
            break
        end
    end

    # Substitute identity matrices into projected indices
    for (n, p) in enumerate(prj)
        if Base.only(p) == 0
            continue
        end
        tensor = zeros(T, linkdims[n], localdims_[n], linkdims[n + 1])
        tensor[:, Base.only(p), :] .= Matrix{T}(LA.I, linkdims[n], linkdims[n + 1])
        sitetensors[n] = tensor
    end

    fulltt = TensorTrain{T,3}(sitetensors)
    return ProjTensorTrain{T}(fulltt, prj)
end

function _fuse(sitedims, p)
    return all(p .> 0) ? _lineari(sitedims, p) : 0
end

# Create a path for a tree
function createpath(proj::Projector, po::PatchOrdering)::Vector{Int}
    key = [_fuse(proj.sitedims[po[n]], proj[po[n]]) for n in 1:length(proj)]
    firstzero = findfirst(x -> x == 0, key)
    if firstzero === nothing
        return key
    else
        return key[1:(firstzero - 1)]
    end
end

function add!(
    root::TreeNode{V}, obj::ProjectableEvaluator{T}, po::PatchOrdering
) where {V,T}
    return add_value!(root, createpath(obj.projector, po), obj)
end
