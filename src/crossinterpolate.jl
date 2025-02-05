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

#======================================================================
   TCI2 Interpolation of a function
======================================================================#
mutable struct TCI2PatchCreator{T} <: AbstractPatchCreator{T,TensorTrainState{T}}
    f::ProjectableEvaluator{T}
    localdims::Vector{Int}
    projector::Projector
    maxbonddim::Int
    verbosity::Int
    tcikwargs::Dict
    tolerance::Float64
    ninitialpivot::Int
    checkbatchevaluatable::Bool
    loginterval::Int
    initialpivots::Vector{MultiIndex} # Make it to Vector{MMultiIndex}?
    recyclepivots::Bool
end

function Base.show(io::IO, obj::TCI2PatchCreator{T}) where {T}
    return print(io, "patchcreator $(obj.f.projector.data)")
end

function TCI2PatchCreator{T}(obj::TCI2PatchCreator{T})::TCI2PatchCreator{T} where {T}
    return TCI2PatchCreator{T}(
        obj.f,
        obj.localdims,
        obj.projector,
        obj.maxbonddim,
        obj.verbosity,
        obj.tcikwargs,
        obj.tolerance,
        obj.ninitialpivot,
        obj.checkbatchevaluatable,
        obj.loginterval,
        obj.initialpivots,
        obj.recyclepivots,
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
    ntry=10, # QUESTION: what is this for? - Gianluca
    ninitialpivot=5,
    checkbatchevaluatable=false,
    loginterval=10,
    initialpivots=MultiIndex[],
    recyclepivots=false,
)::TCI2PatchCreator{T} where {T}
    #t1 = time_ns()
    if projector === nothing
        projector = Projector([[0] for _ in localdims], [[x] for x in localdims])
    end

    #t2 = time_ns()

    if !(f.projector <= projector)
        f = project(f, projector)
    end
    #t3 = time_ns()

    return TCI2PatchCreator{T}(
        f,
        localdims,
        projector,
        maxbonddim,
        verbosity,
        tcikwargs,
        tolerance,
        ninitialpivot,
        checkbatchevaluatable,
        loginterval,
        initialpivots,
        recyclepivots,
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
    recyclepivots=false,
) where {T}
    ncheckhistory = 3
    ranks, errors = TCI.optimize!(
        tci,
        f;
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

    if recyclepivots
        return PatchCreatorResult{T,TensorTrain{T,3}}(
            TensorTrain(tci),
            TCI.maxbonderror(tci) < tolerance && maxbonddim_hist < maxbonddim,
            _globalpivots(tci),
        )

    else
        return PatchCreatorResult{T,TensorTrain{T,3}}(
            TensorTrain(tci),
            TCI.maxbonderror(tci) < tolerance && maxbonddim_hist < maxbonddim,
        )
    end
end

# Generating global pivots from local ones
function _globalpivots(
    tci::TCI.TensorCI2{T}; onlydiagonal=true
)::Vector{MultiIndex} where {T}
    Isets = tci.Iset
    Jsets = tci.Jset
    L = length(Isets)
    p = Set{MultiIndex}()
    # Pivot matrices
    for bondindex in 1:(L - 1)
        if onlydiagonal
            for (x, y) in zip(Isets[bondindex + 1], Jsets[bondindex])
                push!(p, vcat(x, y))
            end
        else
            for x in Isets[bondindex + 1], y in Jsets[bondindex]
                push!(p, vcat(x, y))
            end
        end
    end
    return collect(p)
end

function createpatch(obj::TCI2PatchCreator{T}) where {T}
    fsubset = _FuncAdapterTCI2Subset(obj.f)

    tci = if isapproxttavailable(obj.f)
        # Construct initial pivots from an approximate TT
        projtt = project(
            reshape(approxtt(obj.f; maxbonddim=obj.maxbonddim), obj.f.sitedims),
            reshape(obj.projector, obj.f.sitedims),
        )
        # Converting a TT to a TCI2 object
        tci = TensorCI2{T}(project_on_subsetsiteinds(projtt); tolerance=1e-14)
        if tci.maxsamplevalue == 0.0
            return PatchCreatorResult{T,TensorTrainState{T}}(nothing, true)
        end
        tci
    else
        initialpivots = MultiIndex[]
        if obj.recyclepivots
            # First patching iteration: random pivots
            if length(fsubset.localdims) == length(obj.localdims)
                initialpivots = union(
                    obj.initialpivots,
                    findinitialpivots(fsubset, fsubset.localdims, obj.ninitialpivot),
                )
                # Next iterations: recycle previously generated pivots
            else
                initialpivots = copy(obj.initialpivots)
            end
        else
            initialpivots = union(
                obj.initialpivots,
                findinitialpivots(fsubset, fsubset.localdims, obj.ninitialpivot),
            )
        end

        if all(fsubset.(initialpivots) .== 0)
            return PatchCreatorResult{T,TensorTrainState{T}}(nothing, true)
        end
        TensorCI2{T}(fsubset, fsubset.localdims, initialpivots)
    end

    return _crossinterpolate2!(
        tci,
        fsubset,
        obj.tolerance;
        maxbonddim=obj.maxbonddim,
        verbosity=obj.verbosity,
        checkbatchevaluatable=obj.checkbatchevaluatable,
        loginterval=obj.loginterval,
        recyclepivots=obj.recyclepivots,
    )
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
    initialpivots=MultiIndex[], # Make it to Vector{MMultiIndex}?
    recyclepivots=false,
)::ProjTTContainer{T} where {T}
    creator = TCI2PatchCreator(
        T,
        f,
        collect(prod.(f.sitedims));
        maxbonddim,
        tolerance,
        verbosity,
        ntry=10,
        initialpivots=initialpivots,
        recyclepivots=recyclepivots,
    )
    tmp = adaptiveinterpolate(creator, pordering; verbosity)
    return reshape(tmp, f.sitedims)
end
