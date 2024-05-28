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
    prjmsk = [!isprojectedat(f.projector, n) for n in 1:length(f)]
    return _FuncAdapterTCI2Subset(f, f.sitedims[prjmsk], collect(prod.(f.sitedims[prjmsk])))
end

Base.length(obj::_FuncAdapterTCI2Subset) = length(obj.localdims)

function (obj::_FuncAdapterTCI2Subset)(indexset::MultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end
function (obj::_FuncAdapterTCI2Subset)(indexset::MMultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end

function (obj::_FuncAdapterTCI2Subset{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    orgL = length(obj.f)
    leftindexset_fulllen = fulllength_leftindexset(obj.f.projector, leftindexset)
    rightindexset_fulllen = fulllength_rightindexset(obj.f.projector, rightindexset)

    NL = length(leftindexset_fulllen[1])
    NR = length(rightindexset_fulllen[1])
    M_ = length(obj.f) - NL - NR
    projected = [
        isprojectedat(obj.f.projector, n) ? 1 : Colon() for n in (NL + 1):(orgL - NR)
    ]
    res = batchevaluateprj(obj.f, leftindexset_fulllen, rightindexset_fulllen, Val(M_))
    return res[:, projected..., :]
end

"""
leftindexset: Vector of indices on unprojected indices
Returns: Vector of indices on projected and unprojected indices
"""
function fulllength_leftindexset(
    projector::Projector, leftindexset::AbstractVector{MultiIndex}
)::Vector{MultiIndex}
    if length(leftindexset[1]) == 0
        return fulllength_leftindexset_len0(projector, leftindexset)
    end

    c = 0
    fulllength = 0
    mapping = Vector{Int}(undef, length(leftindexset[1]))
    for n in 1:length(projector)
        if !isprojectedat(projector, n)
            #if c + 1 <= length(mapping)
                mapping[c + 1] = n
            #end
            c += 1
        end
        if c == length(leftindexset[1])
            fulllength = n
            break
        end
    end

    tmp = Int[_lineari(projector.sitedims[n], projector[n]) for n in 1:fulllength]
    leftindexset_ = [deepcopy(tmp) for _ in leftindexset]

    for il in 1:length(leftindexset_)
        leftindexset_[il][mapping] .= leftindexset[il]
    end

    return leftindexset_
end

function fulllength_leftindexset_len0(
    projector::Projector, leftindexset::AbstractVector{MultiIndex}
)::Vector{MultiIndex}
    leftindexset == [Int[]] || error("Invalid leftindexset")
    firstunprojected = findfirst(n->!isprojectedat(projector, n), (n for n in 1:length(projector)))
    endp = firstunprojected === nothing ? length(projector) : firstunprojected - 1
    return [lineari(projector.sitedims[1:endp], projector.data[1:endp])]
end

function fulllength_rightindexset(
    projector::Projector, rightindexset::AbstractVector{MultiIndex}
)
    r = fulllength_leftindexset(reverse(projector), reverse.(rightindexset))
    return collect(reverse.(r))
end

function _reconst_prefix(projector::Projector, pordering::PatchOrdering)
    np = Base.sum((isprojectedat(projector, n) for n in 1:length(projector)))
    return [Base.only(projector[n]) for n in pordering.ordering[1:np]]
end

function adaptiveinterpolate(
    creator::AbstractPatchCreator{T,M},
    pordering::PatchOrdering;
    sleep_time::Float64=1e-6,
    maxnleaves=typemax(Int),
    verbosity=0,
)::Dict{Projector,M} where {T,M}
    leaves = Dict{AbstractPatchCreator,Union{Task,PatchCreatorResult{T,M}}}()

    # Add root
    leaves[creator] = createpatch(creator)

    while true
        sleep(sleep_time) # Not to run the loop too quickly

        done = true
        newtasks = Dict{AbstractPatchCreator{T,M},Task}()
        for (pcreator, leaf) in leaves
            # If the task is done, fetch the result, which
            # will be analyzed in the next loop.
            prefix::Vector{Int} = _reconst_prefix(pcreator.projector, pordering)
            if leaf isa Task
                if istaskdone(leaf)
                    if verbosity > 0
                        println("Fetching a task for $(prefix) ...")
                    end
                    fetched = fetch(leaf)
                    if fetched isa RemoteException
                        err_msg = sprint(showerror, fetched.captured)
                        error("Error in creating a patch for $(prefix): $err_msg")
                    end
                    leaves[pcreator] = fetched
                end
                done = false
                continue
            end

            @assert leaf isa PatchCreatorResult{T,M}

            if !leaf.isconverged && length(leaves) < maxnleaves
                done = false
                delete!(leaves, pcreator)

                for ic in 1:creator.localdims[pordering.ordering[length(prefix) + 1]]
                    prefix_ = vcat(prefix, ic)
                    projector_ = makeproj(pordering, prefix_, pcreator.localdims)
                    pcreator_child = project(pcreator, projector_)

                    if verbosity > 0
                        println("Creating a task for $(prefix_) ...")
                    end
                    t = @task fetch(@spawnat :any createpatch(pcreator_child))
                    newtasks[pcreator_child] = t
                    schedule(t)
                end
            end
        end

        if done
            @assert length(newtasks) == 0
            break
        end

        for (k, v) in newtasks
            leaves[k] = v
        end
    end

    leaves_done = Dict{Projector,M}()
    for (k, v) in leaves
        leaves_done[k.f.projector] =
            isnothing(v.data) ? _zerott(T, k, pordering, creator.localdims) : v.data
    end

    return leaves_done
    #ProjTensorTrainSet(leaves_done, pordering)
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
    rtol::Float64
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

function TCI2PatchCreator{T}(obj::TCI2PatchCreator{T})::TCI2PatchCreator{T} where {T}
    return TCI2PatchCreator{T}(
        obj.f,
        obj.localdims,
        obj.projector,
        obj.rtol,
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
    f,
    localdims::Vector{Int},
    projector::Union{Projector,Nothing}=nothing;
    rtol::Float64=1e-8,
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
        rtol,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        rtol * maxval,
        ninitialpivot,
        checkbatchevaluatable,
        loginterval,
        initialpivots,
    )
end

function _crossinterpolate2(
    ::Type{T},
    f,
    localdims::Vector{Int},
    initialpivots::Vector{MultiIndex},
    tolerance::Float64;
    maxbonddim::Int=typemax(Int),
    verbosity::Int=0,
    checkbatchevaluatable=false,
    loginterval=10,
) where {T}
    ncheckhistory = 3
    tci, others = TCI.crossinterpolate2(
        T,
        f,
        localdims,
        initialpivots;
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

    ncheckhistory_ = min(ncheckhistory, length(others))
    maxbonddim_hist = maximum(others[(end - ncheckhistory_ + 1):end])

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

    return _crossinterpolate2(
        T,
        fsubset,
        fsubset.localdims,
        initialpivots,
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
)::ProjTensorTrain{T,3} where {T}
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
    return ProjTensorTrain{T,3}(fulltt, prj)
end
