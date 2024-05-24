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

function project(
    obj::ProjectableEvaluator{T}, prj::Projector
)::ProjectableEvaluator{T} where {T}
    return error("Must be implemented!")
end


"""
This is similar to batch evaluation.
The difference is as follows.
If some of `M` central indices are projected, the evaluation is done on the projected indices.
The sizes of the correponding indices in the returned array are set to 1.
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
    L = legnth(obj)

    results = zeros(T, (prod(s) for s in sitedims[NL+1:L-NR])...)

    slice = (isproject(obj.projector, n) ? _lineari(sitedims, projector[n]) : Colon() for n in NL+1:L-NR)
    results .= obj(leftindexset, rightindexset, Val(M))[slice...]
    return results
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

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [multii(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [multii(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]

    return obj(leftindexset_, rightindexset_, Val(M))
end

"""
Adapter for a ProjectableEvaluator object:
`f` is a function that can be evaluated at indices (including projected and non-projected indices).

The wrapped function can be evaluated at unprojected indices, and accepts fused indices.
"""
struct ProjectableEvaluatorSubset{T}
    f::ProjectableEvaluator{T}
end

function (obj::ProjectableEvaluatorSubset)(indexset::MultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end
function (obj::ProjectableEvaluatorSubset)(indexset::MMultiIndex)
    return obj.f(fullindices(obj.f.projector, indexset))
end