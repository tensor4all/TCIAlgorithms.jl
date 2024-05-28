"""
Represents an object that can be projected on a subset of indices

Attributes:
- projector: Projector object
- sitedims: Vector{Vector{Int}} of the dimensions of the local indices
"""
abstract type ProjectableEvaluator{T} <: TCI.BatchEvaluator{T} end

# To be implemented reshape

function Base.show(io::IO, obj::ProjectableEvaluator{T}) where {T}
    return print(
        io, "$(typeof(obj)), sitedims: $(obj.sitedims), projector: $(obj.projector.data)"
    )
end

function sum(obj::ProjectableEvaluator{T})::T where {T}
    error("Must be implemented!")
    return zero(T)
end

"""
Project the object on the overlap of `prj` and `obj.projector`.

The requirement for the implementation is that
the projector of the returned object is a subset of `prj`.
"""
function project(
    obj::ProjectableEvaluator{T}, prj::Projector; kwargs...
)::ProjectableEvaluator{T} where {T}
    return error("Must be implemented!")
end

# Override this function
function (obj::ProjectableEvaluator{T})(indexset::MMultiIndex)::T where {T}
    return error("Must be implemented!")
end

"""
Please override this funciton

This is similar to batch evaluation.
The difference is as follows.
If some of `M` central indices are projected, the evaluation is done on the projected indices.
The sizes of the correponding indices in the returned array are set to 1.

`leftindexset` and `rightindexset` are defined for unprojected and projected indices.
"""
function batchevaluateprj(
    obj::ProjectableEvaluator{T},
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    # Please override this funciton
    return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
end

"""
This is similar to `batchevaluateprj`, but the evaluation is done on all `M` indices.
In the returned array, the element evaluates to 0 for a indexset that is out of the projector.
"""
function (obj::ProjectableEvaluator{T})(
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    L = length(obj)

    results_multii = zeros(
        T,
        length(leftindexset),
        Iterators.flatten(obj.sitedims[(NL + 1):(L - NR)])...,
        length(rightindexset),
    )
    slice = map(
        x -> x == 0 ? Colon() : x,
        Iterators.flatten(obj.projector[n] for n in (NL + 1):(L - NR)),
    )
    results_multii[:, slice..., :] .= batchevaluateprj(
        obj, leftindexset, rightindexset, Val(M)
    )
    return reshape(
        results_multii,
        length(leftindexset),
        prod.(obj.sitedims[(NL + 1):(L - NR)])...,
        length(rightindexset),
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

function _multii(obj::ProjectableEvaluator, leftindexset, rightindexset)
    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [multii(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [multii(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]
    return leftindexset_, rightindexset_
end

function _lineari(obj::ProjectableEvaluator, leftindexset, rightindexset)
    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]
    return leftindexset_, rightindexset_
end

# Signe-site-index version
function batchevaluateprj(
    obj::ProjectableEvaluator{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    leftindexset_, rightindexset_ = _multii(obj, leftindexset, rightindexset)
    return batchevaluateprj(obj, leftindexset_, rightindexset_, Val(M))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(indexset::MultiIndex)::T where {T}
    return obj(multii(obj.sitedims, indexset))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    leftindexset_, rightindexset_ = _multii(obj, leftindexset, rightindexset)
    return obj(leftindexset_, rightindexset_, Val(M))
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
        return new{T}(f, sitedims, projector)
    end
    function ProjectableEvaluatorAdapter{T}(
        f::TCI.BatchEvaluator{T}, sitedims::Vector{Vector{Int}}
    ) where {T}
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
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    lmask = [isleftindexset_contained(obj.projector, x) for x in leftindexset]
    rmask = [isrightindexset_contained(obj.projector, x) for x in rightindexset]
    leftindexset_ = [collect(Base.only.(x)) for x in leftindexset[lmask]]
    rightindexset_ = [collect(Base.only.(x)) for x in rightindexset[rmask]]

    result_lrmask = obj.f(leftindexset_, rightindexset_, Val(M))

    # Some of indices might be projected
    NL = length(leftindexset[1])
    NR = length(rightindexset[1])

    NL + NR + M == length(obj) || error("Length mismatch NL: $NL, NR: $NR, M: $M, L: $(length(obj))")

    L = length(obj)
    result::Array{T,M+2} = zeros(
        T,
        length(leftindexset),
        prod.(obj.sitedims[(1 + NL):(L - NR)])...,
        length(rightindexset),
    )
    result[lmask, .., rmask] .= begin
        result_lrmask_multii = reshape(
            result_lrmask,
            size(result_lrmask)[1],
            collect(Iterators.flatten(obj.sitedims[(1 + NL):(L - NR)]))...,
            size(result_lrmask)[end],
        )
        projmask = map(
            p -> p == 0 ? Colon() : p,
            Iterators.flatten(obj.projector[n] for n in (1 + NL):(length(obj) - NR)),
        )
        result_lrmask_multii[:, projmask..., :]
    end
    return result
end

function project(
    obj::ProjectableEvaluatorAdapter{T}, prj::Projector
)::ProjectableEvaluator{T} where {T}
    return ProjectableEvaluatorAdapter{T}(obj.f, obj.sitedims, deepcopy(prj))
end

function fulltensor(obj::ProjectableEvaluator)
    localdims = collect(prod.(obj.sitedims))
    r = [obj(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))]
    returnsize = collect(Iterators.flatten(obj.sitedims))
    return reshape(r, returnsize...)
end
