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
end

"""
# For TCI2
`f` is a function that can be evaluated at full-length indices (including projected and non-projected indices). The wrapped function can be evaluated only on unprojected indices.
"""
struct _FuncAdapterTCI2Subset{T} <: TCI.BatchEvaluator{T}
    f::ProjectableEvaluator{T}
    sitedims::Vector{Vector{Int}}
    localdims::Vector{Int}
end

function _FuncAdapterTCI2Subset(f::ProjectableEvaluator{T}) where {T}
    prjmsk = [!isprojectedat(f.projector, n) for n in 1:length(f.sitedims)]
    localdims = collect(prod.(f.sitedims[prjmsk]))
    return _FuncAdapterTCI2Subset(f, f.sitedims[prjmsk], localdims)
end

Base.length(obj::_FuncAdapterTCI2Subset) = length(obj.localdims)

function (obj::_FuncAdapterTCI2Subset)(indexset::MultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end
function (obj::_FuncAdapterTCI2Subset)(indexset::MMultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end

function (obj::_FuncAdapterTCI2Subset{T})(
    leftmmultiidxset::AbstractVector{MultiIndex},
    rightmmultiidxset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    orgL = length(obj.f.sitedims)
    leftmmultiidxset_fulllen = fulllength_leftmmultiidxset(
        obj.f.projector, leftmmultiidxset
    )
    rightmmultiidxset_fulllen = fulllength_rightmmultiidxset(
        obj.f.projector, rightmmultiidxset
    )

    NL = length(leftmmultiidxset_fulllen[1])
    NR = length(rightmmultiidxset_fulllen[1])
    M_ = length(obj.f.sitedims) - NL - NR
    projected = [
        isprojectedat(obj.f.projector, n) ? 1 : Colon() for n in (NL + 1):(orgL - NR)
    ]
    res = batchevaluateprj(
        obj.f, leftmmultiidxset_fulllen, rightmmultiidxset_fulllen, Val(M_)
    )
    return res[:, projected..., :]
end

"""
leftmmultiidxset: Vector of indices on unprojected indices
Returns: Vector of indices on projected and unprojected indices
"""
function fulllength_leftmmultiidxset(
    projector::Projector, leftmmultiidxset::AbstractVector{MultiIndex}
)::Vector{MultiIndex}
    if length(leftmmultiidxset[1]) == 0
        return fulllength_leftmmultiidxset_len0(projector, leftmmultiidxset)
    end

    c = 0
    fulllength = 0
    mapping = Vector{Int}(undef, length(leftmmultiidxset[1]))
    for n in 1:length(projector)
        if !isprojectedat(projector, n)
            #if c + 1 <= length(mapping)
            mapping[c + 1] = n
            #end
            c += 1
        end
        if c == length(leftmmultiidxset[1])
            fulllength = n
            break
        end
    end

    tmp = Int[_lineari(projector.sitedims[n], projector[n]) for n in 1:fulllength]
    leftmmultiidxset_ = [deepcopy(tmp) for _ in leftmmultiidxset]

    for il in 1:length(leftmmultiidxset_)
        leftmmultiidxset_[il][mapping] .= leftmmultiidxset[il]
    end

    return leftmmultiidxset_
end

function fulllength_leftmmultiidxset_len0(
    projector::Projector, leftmmultiidxset::AbstractVector{MultiIndex}
)::Vector{MultiIndex}
    leftmmultiidxset == [Int[]] || error("Invalid leftmmultiidxset")
    firstunprojected = findfirst(
        n -> !isprojectedat(projector, n), (n for n in 1:length(projector))
    )
    endp = firstunprojected === nothing ? length(projector) : firstunprojected - 1
    return [lineari(projector.sitedims[1:endp], projector.data[1:endp])]
end

function fulllength_rightmmultiidxset(
    projector::Projector, rightmmultiidxset::AbstractVector{MultiIndex}
)
    r = fulllength_leftmmultiidxset(reverse(projector), reverse.(rightmmultiidxset))
    return collect(reverse.(r))
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
            #if verbosity > 0
            ##println("Creating a task for $(prefix_) ...")
            #end
            push!(newtasks, project(creator, projector_))
        end
        return nothing, newtasks
    end
end

function _zerott(T, prefix, po::PatchOrdering, localdims::Vector{Int})
    localdims_ = localdims[maskactiveindices(po, length(prefix))]
    return TensorTrain([zeros(T, 1, d, 1) for d in localdims_])
end

#======================================================================
   TCI2 Interpolation of a function
======================================================================#
mutable struct TCI2PatchCreator{T} <: AbstractPatchCreator{T,TensorTrainState{T}}
    f::ProjectableEvaluator{T}
    localdims::Vector{Int}
    projector::Projector
    tolerance::Float64
    maxbonddim::Int
    verbosity::Int
    tcikwargs::Dict
    maxval::Float64
    atol::Float64
    ninitialpivot::Int
    checkbatchevaluatable::Bool
    loginterval::Int
    initialpivots::Vector{MultiIndex}
end

function Base.show(io::IO, obj::TCI2PatchCreator{T}) where {T}
    return print(io, "patchcreator $(obj.f.projector.data)")
end

function TCI2PatchCreator{T}(obj::TCI2PatchCreator{T})::TCI2PatchCreator{T} where {T}
    return TCI2PatchCreator{T}(
        obj.f,
        obj.localdims,
        obj.projector,
        obj.tolerance,
        obj.maxbonddim,
        obj.verbosity,
        obj.tcikwargs,
        obj.maxval,
        obj.atol,
        obj.ninitialpivot,
        obj.checkbatchevaluatable,
        obj.loginterval,
        obj.initialpivots,
    )
end

function TCI2PatchCreator(
    ::Type{T},
    f::ProjectableEvaluator{T},
    localdims::Vector{Int},
    projector::Union{Projector,Nothing}=nothing;
    tolerance::Float64=1e-8,
    maxbonddim::Int=100,
    verbosity::Int=0,
    tcikwargs=Dict(),
    ntry=100,
    ninitialpivot=5,
    checkbatchevaluatable=false,
    loginterval=10,
    initialpivots=Vector{MultiIndex}[],
)::TCI2PatchCreator{T} where {T}
    maxval, _ = _estimate_maxval(f, localdims; ntry=ntry)
    if projector === nothing
        projector = Projector([[0] for _ in localdims], [[x] for x in localdims])
    end

    if !(f.projector <= projector)
        f = project(f, projector)
    end

    return TCI2PatchCreator{T}(
        f,
        localdims,
        projector,
        tolerance,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        tolerance * maxval,
        ninitialpivot,
        checkbatchevaluatable,
        loginterval,
        initialpivots,
    )
end

function TCI2PatchCreator(
    ::Type{T},
    f,
    localdims::Vector{Int},
    projector::Union{Projector,Nothing}=nothing;
    kwargs...,
) where {T}
    return TCI2PatchCreator(
        T, makeprojectable(T, f, localdims), localdims, projector; kwargs...
    )
end

function _crossinterpolate2!(
    tci::TensorCI2{T},
    f,
    tolerance::Float64;
    maxbonddim::Int=typemax(Int),
    verbosity::Int=0,
    checkbatchevaluatable=false,
    loginterval=10,
) where {T}
    ncheckhistory = 3
    ranks, errors = TCI.optimize!(tci, f;
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        verbosity=verbosity,
        normalizeerror=false,
        loginterval=loginterval,
        nsearchglobalpivot=10,
        maxiter=10,
        ncheckhistory=ncheckhistory,
        tolmarginglobalsearch=10.0,
        checkbatchevaluatable=checkbatchevaluatable,
    )
    if maximum(TCI.linkdims(tci)) == 0
        error(
            "TCI has zero rank, maxsamplevalue: $(tci.maxsamplevalue), tolerance: ($tolerance)",
        )
    end

    ncheckhistory_ = min(ncheckhistory, length(errors))
    maxbonddim_hist = maximum(ranks[(end - ncheckhistory_ + 1):end])

    return PatchCreatorResult{T,TensorTrain{T,3}}(
        TensorTrain(tci), TCI.maxbonderror(tci) < tolerance && maxbonddim_hist < maxbonddim
    )
end

function project(
    obj::TCI2PatchCreator{T}, projector::Projector
)::TCI2PatchCreator{T} where {T}
    projector <= obj.projector || error(
        "Projector $projector is not a subset of the original projector $(obj.f.projector)",
    )

    obj_copy = TCI2PatchCreator{T}(obj) # shallow copy
    obj_copy.projector = deepcopy(projector)
    obj_copy.f = project(obj_copy.f, projector)
    return obj_copy
end

function createpatch(obj::TCI2PatchCreator{T}) where {T}
    proj = obj.projector
    fsubset = _FuncAdapterTCI2Subset(obj.f)

    tci = 
    if isapproxttavailable(obj.f)
        # Construct initial pivots from an approximate TT
        projtt = project(
            reshape(approxtt(obj.f; maxbonddim=obj.maxbonddim), obj.f.sitedims),
            reshape(obj.projector, obj.f.sitedims))
        # Converting a TT to a TCI2 object
        TensorCI2{T}(project_on_subsetsiteinds(projtt); tolerance=1e-14)
    else
        # Random initial pivots
        initialpivots = MultiIndex[]
        let
            mask = [!isprojectedat(proj, n) for n in 1:length(proj)]
            for idx in obj.initialpivots
                idx_ = [[i] for i in idx]
                if idx_ <= proj
                    push!(initialpivots, idx[mask])
                end
            end
        end
        append!(initialpivots, findinitialpivots(fsubset, fsubset.localdims, obj.ninitialpivot))
        if all(fsubset.(initialpivots) .== 0)
           return PatchCreatorResult{T,TensorTrainState{T}}(nothing, true)
        end
        TensorCI2{T}(fsubset, fsubset.localdims, initialpivots)
    end

    return _crossinterpolate2!(
        tci,
        fsubset,
        obj.atol;
        maxbonddim=obj.maxbonddim,
        verbosity=obj.verbosity,
        checkbatchevaluatable=obj.checkbatchevaluatable,
        loginterval=obj.loginterval,
    )
end

function _estimate_maxval(f, localdims; ntry=100)
    pivot = fill(1, length(localdims))
    maxval::Float64 = abs(f(pivot))
    for i in 1:ntry
        pivot_ = [rand(1:localdims[i]) for i in eachindex(localdims)]
        pivot_ = TCI.optfirstpivot(f, localdims, pivot_)
        maxval_ = abs(f(pivot_))
        if maxval_ > maxval
            maxval = maxval_
            pivot .= pivot_
        end
    end
    return maxval, pivot
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

function adaptiveinterpolate(
    creator::TCI2PatchCreator{T}, pordering::PatchOrdering; verbosity=0
)::ProjTTContainer{T} where {T}
    queue = TaskQueue{TCI2PatchCreator{T},ProjTensorTrain{T}}([creator])
    results = loop(
        queue, x -> __taskfunc(x, pordering; verbosity=verbosity); verbosity=verbosity
    )
    return ProjTTContainer(results)
end


function adaptiveinterpolate(
    f::ProjectableEvaluator{T},
    pordering::PatchOrdering=PatchOrdering(collect(1:length(f.sitedims)));
    verbosity=0,
    maxbonddim=typemax(Int),
    tolerance=1e-8,
)::ProjTTContainer{T} where {T}
    creator = TCI2PatchCreator(
        T, f, collect(prod.(f.sitedims)); maxbonddim, tolerance, verbosity, ntry=10
    )
    tmp = adaptiveinterpolate(creator, pordering; verbosity)
    return reshape(tmp, f.sitedims)
end

function ProjTensorTrainSet(
    tts::Dict{Vector{Vector{Int}},TensorTrain{T,N}},
    sitedims::AbstractVector{<:AbstractVector{Int}},
    po::PatchOrdering,
)::ProjTensorTrainSet{T} where {T,N}
    keys_ = keys(tts)
    L = Base.only(unique([length(tt) + length(p) for (p, tt) in tts]))
    L == length(sitedims) || error("Inconsistent length")

    globalprojecter = makeproj([fill(0, length(s)) for s in sitedims], sitedims)

    tts_ = ProjTensorTrain{T,N}[]
    for prefix in keys_
        p = makeproj(po, prefix, sitedims)
        push!(tts_, ProjTensorTrain(tts[prefix], sitedims, p))
    end

    return ProjTensorTrainSet(tts_, globalprojecter, sitedims)
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
