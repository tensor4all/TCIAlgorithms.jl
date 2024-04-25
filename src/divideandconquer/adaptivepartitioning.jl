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

function fullindices(
    po::PatchOrdering, prefix::Vector{Vector{Int}}, restindices::Vector{Vector{Int}}
)
    length(prefix) + length(restindices) == length(po.ordering) ||
        error("Inconsistent length")
    res = [Int[] for _ in 1:(length(prefix) + length(restindices))]

    res[po.ordering[1:length(prefix)]] .= prefix
    res[maskactiveindices(po, length(prefix))] .= restindices
    return res
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

function adaptiveinterpolate(
    creator::AbstractPatchCreator{T,M},
    pordering::PatchOrdering;
    sleep_time::Float64=1e-6,
    maxnleaves=100,
    verbosity=0,
)::Dict{Vector{MultiIndex},M} where {T,M}
    leaves = Dict{Vector{Int},Union{Task,PatchCreatorResult{T,M}}}()

    # Add root
    leaves[[]] = createpatch(creator, pordering, Vector{Int}[])

    while true
        sleep(sleep_time) # Not to run the loop too quickly

        done = true
        newtasks = Dict{Vector{Int},Task}()
        for (prefix, leaf) in leaves
            # If the task is done, fetch the result, which
            # will be analyzed in the next loop.
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
                    leaves[prefix] = fetched
                end
                done = false
                continue
            end

            @assert leaf isa PatchCreatorResult{T,M}

            if !leaf.isconverged && length(leaves) < maxnleaves
                done = false
                delete!(leaves, prefix)

                for ic in 1:creator.localdims[pordering.ordering[length(prefix) + 1]]
                    prefix_ = vcat(prefix, ic)
                    if verbosity > 0
                        println("Creating a task for $(prefix_) ...")
                    end
                    t = @task fetch(
                        @spawnat :any createpatch(
                            creator, pordering, [[x] for x in prefix_]
                        )
                    )
                    newtasks[prefix_] = t
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

    leaves_done = Dict{Vector{Vector{Int}},M}()
    for (k, v) in leaves
        leaves_done[[[x] for x in k]] = isnothing(v.data) ? _zerott(T, k, pordering, creator.localdims) : v.data
    end

    return leaves_done
    #PartitionedTensorTrain(leaves_done, pordering)
end

function _zerott(T, prefix, po::PatchOrdering, localdims::Vector{Int})
    localdims_ = localdims[maskactiveindices(po, length(prefix))]

    return TensorTrain([zeros(T, 1, d, 1) for d in localdims_])
end

#======================================================================
   TCI2 Interpolation of a function
======================================================================#
TensorTrainState{T} = TensorTrain{T,3} where {T}
_evaluate(obj::TensorCI2, idx::Vector{Vector{Int}}) = TCI.evaluate(obj, map(first, idx))
function _evaluate(obj::TensorTrainState{T}, idx::AbstractVector{Int}) where {T}
    return TCI.evaluate(obj, idx)
end
function _evaluate(obj::TensorTrainState{T}, idx::Vector{Vector{Int}}) where {T}
    return TCI.evaluate(obj, map(first, idx))
end

mutable struct TCI2PatchCreator{T} <: AbstractPatchCreator{T,TensorTrainState{T}}
    f::Any
    localdims::Vector{Int}
    rtol::Float64
    maxbonddim::Int
    verbosity::Int
    tcikwargs::Dict
    maxval::Float64
    atol::Float64
    ninitialpivot::Int
    checkbatchevaluatable::Bool
end

function TCI2PatchCreator(
    ::Type{T},
    f,
    localdims::Vector{Int};
    rtol::Float64=1e-8,
    maxbonddim::Int=100,
    verbosity::Int=0,
    tcikwargs=Dict(),
    ntry=100,
    ninitialpivot=5,
    checkbatchevaluatable=false
)::TCI2PatchCreator{T} where {T}
    maxval, _ = _estimate_maxval(f, localdims; ntry=ntry)
    return TCI2PatchCreator{T}(
        f,
        localdims,
        rtol,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        rtol * maxval,
        ninitialpivot,
        checkbatchevaluatable
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
    checkbatchevaluatable=false
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
        loginterval=1,
        nsearchglobalpivot=10,
        maxiter=10,
        ncheckhistory=ncheckhistory,
        tolmarginglobalsearch=10.0,
        checkbatchevaluatable=checkbatchevaluatable
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

struct _FuncWrapper{T,F} <: TCI.BatchEvaluator{T}
    f::F
    pordering::PatchOrdering
    prefix::Vector{Vector{Int}}
    fromleft::Bool
end

function _FuncWrapper(::Type{T}, f::F, pordering, prefix) where {T,F}
    all(pordering.ordering[1:end-1] .< pordering.ordering[2:end]) || all(pordering.ordering[1:end-1] .> pordering.ordering[2:end]) ||
        error("Ordering must be increasing or decreasing (just currently implemented)")
    fromleft = all(pordering.ordering[1:end-1] .< pordering.ordering[2:end]) 
    return _FuncWrapper{T,F}(f, pordering, prefix, fromleft)
end

function (obj::_FuncWrapper{T})(indexset::MultiIndex)::T where {T}
    idx = fullindices(obj.pordering, obj.prefix, [[x_] for x_ in indexset])
    return obj.f(map(first, idx))
end

function (obj::_FuncWrapper{T})(leftindexset::Vector{MultiIndex}, rightindexset::Vector{MultiIndex}, ::Val{M})::Array{T,M + 2} where {T,M}
    if obj.fromleft
        p = first.(obj.prefix)
        l::Vector{Vector{Int}} = [vcat(p, indexset) for indexset in leftindexset]
        return obj.f(l, rightindexset, Val(M))
    else
        p = reverse(first.(obj.prefix))
        r::Vector{Vector{Int}} = [vcat(indexset, p) for indexset in rightindexset]
        return obj.f(leftindexset, r, Val(M))
    end
end

function createpatch(
    obj::TCI2PatchCreator{T}, pordering::PatchOrdering, prefix::Vector{Vector{Int}}
) where {T}
    mask = maskactiveindices(pordering, length(prefix))
    localdims_ = obj.localdims[mask]

    function _f(x::Vector{Int})::T
        idx = fullindices(pordering, prefix, [[x_] for x_ in x])
        return obj.f(map(first, idx))
    end
    
    wrapf = obj.f isa TCI.BatchEvaluator ? _FuncWrapper(T, obj.f, pordering, prefix) : _f

    firstpivots = findinitialpivots(wrapf, localdims_, obj.ninitialpivot)

    if all(wrapf.(firstpivots) .== 0)
        return PatchCreatorResult{T,TensorTrainState{T}}(nothing, true)
    end

    return _crossinterpolate2(
        T,
        wrapf,
        localdims_,
        firstpivots,
        obj.atol;
        maxbonddim=obj.maxbonddim,
        verbosity=obj.verbosity,
        checkbatchevaluatable=obj.checkbatchevaluatable
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

function Projector(
    po::PatchOrdering, prefix::Vector{Vector{Int}}, sitedims::Vector{Vector{Int}}
)
    data = Vector{Int}[fill(0, length(s)) for s in sitedims]
    for (i, n) in enumerate(po.ordering[1:length(prefix)])
        data[n] = deepcopy(prefix[i])
    end
    return Projector(data, sitedims)
end

function PartitionedTensorTrain(
    tts::Dict{Vector{Vector{Int}},TensorTrain{T,N}},
    sitedims::AbstractVector{<:AbstractVector{Int}},
    po::PatchOrdering,
)::PartitionedTensorTrain{T} where {T,N}
    keys_ = keys(tts)
    L = Base.only(unique([length(tt) + length(p) for (p, tt) in tts]))
    L == length(sitedims) || error("Inconsistent length")

    globalprojecter = Projector([fill(0, length(s)) for s in sitedims], sitedims)

    tts_ = ProjectedTensorTrain{T,N}[]
    for prefix in keys_
        p = Projector(po, prefix, sitedims)
        push!(tts_, ProjectedTensorTrain(tts[prefix], sitedims, p))
    end

    return PartitionedTensorTrain(tts_, globalprojecter, sitedims)
end

"""
`tt` is a TensorTrain{T,3} and `prj` is a Projector.
`tt` is defined on unprojected indices.
Return a ProjectedTensorTrain{T} defined on full indices.
"""
function ProjectedTensorTrain(
    tt::TensorTrain{T,3}, localdims::AbstractVector{<:AbstractVector{Int}}, prj::Projector
)::ProjectedTensorTrain{T,3} where {T}
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
    return ProjectedTensorTrain{T,3}(fulltt, prj)
end
