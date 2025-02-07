"""
Represents an object that can be projected on a subset of indices

Attributes:
- projector: Projector object
- sitedims: Vector{Vector{Int}} of the dimensions of the local indices
"""
abstract type ProjectableEvaluator{T} <: TCI.BatchEvaluator{T} end

function Base.show(io::IO, obj::ProjectableEvaluator{T}) where {T}
    return print(
        io, "$(typeof(obj)), sitedims: $(obj.sitedims), projector: $(obj.projector.data)"
    )
end

"""
Project the object on the overlap of `prj` and `obj.projector`.

The requirement for the implementation is that
the projector of the returned object is a subset of `prj`.
"""
function project(
    obj::ProjectableEvaluator{T}, prj::Projector; kwargs...
)::ProjectableEvaluator{T} where {T}
    return error("Must be implemented for $(typeof(obj))!")
end

# Override this function
function (obj::ProjectableEvaluator{T})(indexset::MMultiIndex)::T where {T}
    return error("Must be implemented for $(typeof(obj))!")
end

# Override this function
function Base.reshape(
    obj::ProjectableEvaluator{T}, sitedims::AbstractVector{<:AbstractVector{Int}}
)::ProjectableEvaluator{T} where {T}
    return error("Must be implemented for $(typeof(obj))!")
end

"""
Compute a tensor train approximation of the object.
The computation should be quick because the result will be used as initial guesses for the optimization.
Override this function
 """
function approxtt(
    obj::ProjectableEvaluator{T}; maxbonddim=typemax(Int), tolerance=1e-14, kwargs...
)::ProjTensorTrain{T} where {T}
    return error("Must be implemented for $(typeof(obj))!")
end

function isapproxttavailable(obj::ProjectableEvaluator)
    return false
end

"""
Please override this funciton

This is similar to batch evaluation.
The difference is as follows.
If some of `M` central indices are projected, the evaluation is done on the projected indices.
The sizes of the correponding indices in the returned array are set to 1.

`leftmmultiidxset` and `rightmmultiidxset` are defined for unprojected and projected indices.
"""
function batchevaluateprj(
    obj::ProjectableEvaluator{T},
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    # Please override this funciton
    error("Must be implemented!")
    return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
end

"""
This is similar to `batchevaluateprj`, but the evaluation is done on all `M` indices.
In the returned array, the element evaluates to 0 for a indexset that is out of the projector.
"""
function (obj::ProjectableEvaluator{T})(
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])
    L = length(obj.sitedims)

    results_multii = zeros(
        T,
        length(leftmmultiidxset),
        Iterators.flatten(obj.sitedims[(NL + 1):(L - NR)])...,
        length(rightmmultiidxset),
    )
    slice = map(
        x -> x == 0 ? Colon() : x,
        Iterators.flatten(obj.projector[n] for n in (NL + 1):(L - NR)),
    )

    # QUESTION: I fixed the problem of batchevaluate for Containers like this, let me know if 
    # it is okay - Gianluca
    mask = vcat(obj.projector[(NL + 1):(L - NR)]...) .== 0
    sliced_sitedims = vcat(obj.sitedims[(NL + 1):(L - NR)]...)[mask]

    results_multii[:, slice..., :] .= reshape(
        batchevaluateprj(obj, leftmmultiidxset, rightmmultiidxset, Val(M)),
        length(leftmmultiidxset),
        sliced_sitedims...,
        length(rightmmultiidxset),
    )

    return reshape(
        results_multii,
        length(leftmmultiidxset),
        prod.(obj.sitedims[(NL + 1):(L - NR)])...,
        length(rightmmultiidxset),
    )
end

function projector(obj::ProjectableEvaluator{T})::Projector where {T}
    return obj.projector
end

function projector(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.projector[l][ilegg] for l in 1:length(obj)]
end

function sitedims(obj::ProjectableEvaluator{T}, ilegg::Int)::Vector{Int} where {T}
    return [obj.sitedims[l][ilegg] for l in 1:length(obj)]
end

function _multii(obj::ProjectableEvaluator, leftmmultiidxset, rightmmultiidxset)
    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])
    leftmmultiidxset_ = [multii(obj.sitedims[1:NL], x) for x in leftmmultiidxset]
    rightmmultiidxset_ = [
        multii(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset
    ]
    return leftmmultiidxset_, rightmmultiidxset_
end

function _lineari(obj::ProjectableEvaluator, leftmmultiidxset, rightmmultiidxset)
    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])
    leftmmultiidxset_ = [lineari(obj.sitedims[1:NL], x) for x in leftmmultiidxset]
    rightmmultiidxset_ = [
        lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset
    ]
    return leftmmultiidxset_, rightmmultiidxset_
end

# Single-site-index version
function batchevaluateprj(
    obj::ProjectableEvaluator{T},
    leftmultiidxset::AbstractVector{MultiIndex},
    rightmultiidxset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    M >= 0 || error("The order of the result must be non-negative")
    if length(leftmultiidxset) * length(rightmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    leftmmultiidxset_, rightmmultiidxset_ = _multii(obj, leftmultiidxset, rightmultiidxset)
    return batchevaluateprj(obj, leftmmultiidxset_, rightmmultiidxset_, Val(M))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(indexset::MultiIndex)::T where {T}
    return obj(multii(obj.sitedims, indexset))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(
    leftmmultiidxset::AbstractVector{MultiIndex},
    rightmmultiidxset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    leftmmultiidxset_, rightmmultiidxset_ = _multii(
        obj, leftmmultiidxset, rightmmultiidxset
    )
    return obj(leftmmultiidxset_, rightmmultiidxset_, Val(M))
end

"""
Convert function `f` to a ProjectableEvaluator object
"""
struct ProjectableEvaluatorAdapter{T} <: ProjectableEvaluator{T}
    f::TCI.BatchEvaluator{T}
    sitedims::Vector{Vector{Int}}
    projector::Projector
    function ProjectableEvaluatorAdapter{T}(
        f::TCI.BatchEvaluator{T}, projector::Projector
    ) where {T}
        return new{T}(f, projector.sitedims, projector)
    end
    function ProjectableEvaluatorAdapter{T}(
        f::TCI.BatchEvaluator{T}, sitedims::Vector{Vector{Int}}, projector::Projector
    ) where {T}
        length(vcat(sitedims...)) == length(sitedims) ||
            error("No sitedims grouping allowed")
        return new{T}(f, sitedims, reshape(projector, sitedims))
    end
    function ProjectableEvaluatorAdapter{T}(
        f::TCI.BatchEvaluator{T}, sitedims::Vector{Vector{Int}}
    ) where {T}
        length(vcat(sitedims...)) == length(sitedims) ||
            error("No sitedims grouping allowed")
        return new{T}(f, sitedims, Projector([[0] for _ in sitedims], sitedims))
    end
end

Base.length(obj::ProjectableEvaluatorAdapter) = length(obj.sitedims)

function makeprojectable(::Type{T}, f::Function, localdims::Vector{Int}) where {T}
    return ProjectableEvaluatorAdapter{T}(
        f isa TCI.BatchEvaluator ? f : TCI.makebatchevaluatable(T, f, localdims),
        [[x] for x in localdims],
    )
end

function (obj::ProjectableEvaluatorAdapter{T})(indexset::MMultiIndex)::T where {T}
    return indexset <= obj.projector ? obj.f(lineari(obj.sitedims, indexset)) : zero(T)
end

function batchevaluateprj(
    obj::ProjectableEvaluatorAdapter{T},
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    lmask = [isleftmmultiidx_contained(obj.projector, x) for x in leftmmultiidxset]
    rmask = [isrightmmultiidx_contained(obj.projector, x) for x in rightmmultiidxset]
    leftmmultiidxset_ = [collect(Base.only.(x)) for x in leftmmultiidxset[lmask]]
    rightmmultiidxset_ = [collect(Base.only.(x)) for x in rightmmultiidxset[rmask]]

    result_lrmask = obj.f(leftmmultiidxset_, rightmmultiidxset_, Val(M))

    # Some of indices might be projected
    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])
    L = length(obj)

    NL + NR + M == L || error("Length mismatch NL: $NL, NR: $NR, M: $M, L: $(L)")

    returnshape = projectedshape(obj.projector, NL + 1, L - NR)
    result::Array{T,M + 2} = zeros(
        T, length(leftmmultiidxset), returnshape..., length(rightmmultiidxset)
    )

    projmask = map(
        p -> p == 0 ? Colon() : p,
        Iterators.flatten(obj.projector[n] for n in (1 + NL):(L - NR)),
    )
    slice = map(
        p -> p == 0 ? Colon() : 1,
        Iterators.flatten(obj.projector[n] for n in (1 + NL):(L - NR)),
    )

    result[lmask, slice..., rmask] .= begin
        result_lrmask_multii = reshape(
            result_lrmask,
            size(result_lrmask)[1],
            collect(Iterators.flatten(obj.sitedims[(1 + NL):(L - NR)]))...,
            size(result_lrmask)[end],
        )        # Gianluca - this step might be not needed. I leave it for safety 
        result_lrmask_multii[:, projmask..., :]
    end
    return result
end

function project(
    obj::ProjectableEvaluatorAdapter{T}, prj::Projector
)::ProjectableEvaluator{T} where {T}
    prj <= obj.projector || error("Projection incompatible with $(obj.projector.data)")
    return ProjectableEvaluatorAdapter{T}(obj.f, obj.sitedims, deepcopy(prj))
end

"""
Evaluate `obj` at all possible indexsets and return a full tensor
"""
function fulltensor(obj::ProjectableEvaluator{T}; fused::Bool=false)::Array{T} where {T}
    localdims = collect(prod.(obj.sitedims))
    r = [obj(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
    if fused
        returnsize = collect(prod.(obj.sitedims))
    else
        returnsize = collect(Iterators.flatten(obj.sitedims))
    end
    return reshape(r, returnsize...)
end
