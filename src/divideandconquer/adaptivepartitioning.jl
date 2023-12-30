"""
Specify the ordering of patching
"""
struct PatchOrdering
    ordering::Vector{Int}
    function PatchOrdering(ordering::Vector{Int})
        sort(ordering) == collect(1:length(ordering)) || error("Inconsistent ordering")
        new(ordering)
    end
end

"""
n is the length of the prefix.
"""
function maskactiveindices(po::PatchOrdering, nprefix::Int)
    mask = ones(Bool, length(po.ordering))
    mask[po.ordering[1:nprefix]] .= false
    return mask
end

function fullindices(
    po::PatchOrdering,
    prefix::Vector{Vector{Int}},
    restindices::Vector{Vector{Int}},
)
    length(prefix) + length(restindices) == length(po.ordering) ||
        error("Inconsistent length")
    res = [Int[] for _ = 1:(length(prefix)+length(restindices))]

    res[po.ordering[1:length(prefix)]] .= prefix
    res[maskactiveindices(po, length(prefix))] .= restindices
    return res
end

#==
abstract type AbstractAdaptiveTCINode{C} end

struct AdaptiveLeaf{C} <: AbstractAdaptiveTCINode{C}
    data::C
    prefix::Vector{Vector{Int}}
    pordering::PatchOrdering
end

function Base.show(io::IO, obj::AdaptiveLeaf{C}) where {C}
    prefix = prod(["$x" for x in obj.prefix])
    return println(
        io,
        "  "^length(prefix) * "Leaf $(prefix): rank=$(maximum(_linkdims(obj.data)))",
    )
end

_linkdims(tci::TensorCI2{T}) where {T} = TCI.linkdims(tci)
_linkdims(tt::TensorTrain{T,N}) where {T,N} =
    [last(size(tt.T[n])) for n = 1:(length(tt.T)-1)]

struct AdaptiveInternalNode{C} <: AbstractAdaptiveTCINode{C}
    children::Dict{Vector{Int},AbstractAdaptiveTCINode{C}}
    prefix::Vector{Vector{Int}}
    pordering::PatchOrdering

    function AdaptiveInternalNode{C}(
        children::Dict{Vector{Int},AbstractAdaptiveTCINode{C}},
        prefix::Vector{Vector{Int}},
        pordering::PatchOrdering,
    ) where {C}
        return new{C}(children, prefix, pordering)
    end
end

"""
prefix is the common prefix of all children
"""
function AdaptiveInternalNode{C}(
    children::Vector{AbstractAdaptiveTCINode{C}},
    prefix::Vector{Vector{Int}},
    pordering::PatchOrdering,
) where {C}
    d = Dict{Vector{Int},AbstractAdaptiveTCINode{C}}()
    for child in children
        d[child.prefix[end]] = child
    end
    return AdaptiveInternalNode{C}(d, prefix, pordering)
end

function Base.show(io::IO, obj::AdaptiveInternalNode{C}) where {C}
    println(
        io,
        "  "^length(obj.prefix) *
        "InternalNode $(obj.prefix) with $(length(obj.children)) children",
    )
    for (k, v) in obj.children
        Base.show(io, v)
    end
end

"""
Evaluate the tree at given idx
"""
function evaluate(
    obj::AdaptiveInternalNode{C},
    idx::AbstractVector{T},
) where {C,T<:AbstractArray{Int}}
    child_key = idx[obj.pordering.ordering[length(obj.prefix)+1]]
    return evaluate(obj.children[child_key], idx)
end

function _onlyactiveindices(
    obj::AbstractAdaptiveTCINode{C},
    idx::AbstractVector{T},
) where {C,T<:AbstractArray{Int}}
    return idx[maskactiveindices(obj.pordering, length(obj.prefix))]
end

function evaluate(
    obj::AdaptiveLeaf{C},
    idx::AbstractVector{T},
) where {C,T<:AbstractArray{Int}}
    return _evaluate(obj.data, _onlyactiveindices(obj, idx))
end


"""
Convert a dictionary of patches to a tree
"""
function _to_tree(
    patches::Dict{Vector{Vector{Int}},C},
    pordering::PatchOrdering;
    nprefix = 0,
)::AbstractAdaptiveTCINode{C} where {C}
    length(unique(k[1:nprefix] for (k, v) in patches)) == 1 ||
        error("Inconsistent prefixes")

    common_prefix = first(patches)[1][1:nprefix]

    # Return a leaf
    if nprefix == length(first(patches)[1])
        return AdaptiveLeaf{C}(first(patches)[2], common_prefix, pordering)
    end

    subgroups = Dict{Vector{Int},Dict{Vector{Vector{Int}},C}}()

    # Look at the first index after nprefix skips
    # and group the patches by that index
    for (k, v) in patches
        idx = k[nprefix+1]
        if idx in keys(subgroups)
            subgroups[idx][k] = v
        else
            subgroups[idx] = Dict{Vector{Vector{Int}},C}(k => v)
        end
    end

    # Recursively construct the tree
    children = AbstractAdaptiveTCINode{C}[]
    for (_, grp) in subgroups
        push!(children, _to_tree(grp, pordering; nprefix = nprefix + 1))
    end

    return AdaptiveInternalNode{C}(children, common_prefix, pordering)
end
==#

"""
T: Float64, ComplexF64, etc.
M: TensorCI2, MPS, etc.
"""
abstract type AbstractPatchCreator{T,M} end


mutable struct PatchCreatorResult{T,M}
    data::M
    isconverged::Bool
end


function adaptivepartion(
    creator::AbstractPatchCreator{T,M},
    pordering::PatchOrdering;
    sleep_time::Float64 = 1e-6,
    maxnleaves = 100,
    verbosity = 0,
)::Dict{MultiIndex,} where {T,M}
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
                    leaves[prefix] = fetch(leaf)
                end
                done = false
                continue
            end

            @assert leaf isa PatchCreatorResult{T,M}

            if !leaf.isconverged && length(leaves) < maxnleaves
                done = false
                delete!(leaves, prefix)

                for ic = 1:creator.localdims[pordering.ordering[length(prefix)+1]]
                    prefix_ = vcat(prefix, ic)
                    if verbosity > 0
                        println("Creating a task for $(prefix_) ...")
                    end
                    t = @task fetch(@spawnat :any createpatch(creator, pordering, [[x] for x in prefix_]))
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
        leaves_done[[[x] for x in k]] = v.data
    end

    return leaves_done
end


#======================================================================
   TCI2 Interpolation of a function
======================================================================#
TensorTrainState{T} = TensorTrain{T,3} where {T}
_evaluate(obj::TensorCI2, idx::Vector{Vector{Int}}) = TCI.evaluate(obj, map(first, idx))
_evaluate(obj::TensorTrainState{T}, idx::AbstractVector{Int}) where {T} =
    TCI.evaluate(obj, idx)
_evaluate(obj::TensorTrainState{T}, idx::Vector{Vector{Int}}) where {T} =
    TCI.evaluate(obj, map(first, idx))

mutable struct TCI2PatchCreator{T} <: AbstractPatchCreator{T,TensorTrainState{T}}
    f::Any
    localdims::Vector{Int}
    rtol::Float64
    maxbonddim::Int
    verbosity::Int
    tcikwargs::Dict
    maxval::Float64
    atol::Float64
end


function TCI2PatchCreator(
    ::Type{T},
    f,
    localdims::Vector{Int};
    rtol::Float64 = 1e-8,
    maxbonddim::Int = 100,
    verbosity::Int = 0,
    tcikwargs = Dict(),
    ntry = 100,
)::TCI2PatchCreator{T} where {T}
    maxval, _ = _estimate_maxval(f, localdims; ntry = ntry)
    return TCI2PatchCreator{T}(
        f,
        localdims,
        rtol,
        maxbonddim,
        verbosity,
        tcikwargs,
        maxval,
        rtol * maxval,
    )
end


function _crossinterpolate2(
    ::Type{T},
    f,
    localdims::Vector{Int},
    initialpivots::Vector{MultiIndex},
    tolerance::Float64;
    maxbonddim::Int = typemax(Int),
    verbosity::Int = 0,
) where {T}
    tci, others = TCI.crossinterpolate2(
        T,
        f,
        localdims,
        initialpivots;
        tolerance = 1e-1 * tolerance,
        maxbonddim = maxbonddim,
        verbosity = verbosity,
        normalizeerror = false,
    )

    err(x) = abs(TCI.evaluate(tci, x) - f(x))
    abserror = 0.0
    for _ = 1:10
        p = TCI.optfirstpivot(err, localdims, [rand(1:d) for d in localdims])
        newerr = abs(err(p))
        if abserror < newerr
            abserror = newerr
        end
    end

    true_error = max(TCI.maxbonderror(tci), abserror)

    return PatchCreatorResult{T,TensorTrain{T,3}}(
        TensorTrain(tci),
        true_error < tolerance && maximum(TCI.linkdims(tci)) <= maxbonddim,
    )
end

function createpatch(
    obj::TCI2PatchCreator{T},
    pordering::PatchOrdering,
    prefix::Vector{Vector{Int}},
) where {T}
    mask = maskactiveindices(pordering, length(prefix))
    localdims_ = obj.localdims[mask]
    #f_ = x -> obj.f(fullindices(pordering, prefix, x))

    function f_(x::Vector{Int})::T
        idx = fullindices(pordering, prefix, [[x_] for x_ in x])
        return obj.f(map(first, idx))
    end

    firstpivot = TCI.optfirstpivot(f_, localdims_, fill(1, length(localdims_)))

    return _crossinterpolate2(
        T,
        f_,
        localdims_,
        [firstpivot],
        obj.atol;
        maxbonddim = obj.maxbonddim,
        verbosity = obj.verbosity,
    )
end


function _estimate_maxval(f, localdims; ntry = 100)
    pivot = fill(1, length(localdims))
    maxval::Float64 = abs(f(pivot))
    for i = 1:ntry
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
