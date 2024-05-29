"""
Lazy evaluation for matrix multiplication of two TTOs
Two site indices on each site.
"""
struct LazyMatrixMul{T} <: ProjectableEvaluator{T}
    coeff::T
    contraction::TCI.Contraction{T}
    projector::Projector
    sitedims::Vector{Vector{Int}}
    a::ProjTensorTrain{T}
    b::ProjTensorTrain{T}
end

function LazyMatrixMul{T}(
    a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(T)
) where {T}
    # This restriction is due to simulicity and to be removed.
    all(length.(a.sitedims) .== 2) || error("The number of site indices must be 2")
    all(length.(b.sitedims) .== 2) || error("The number of site indices must be 2")
    sitedims_ab = [[x[1], y[2]] for (x, y) in zip(a.sitedims, b.sitedims)]
    a_tt = TensorTrain{T,4}(a.data, a.sitedims)
    b_tt = TensorTrain{T,4}(b.data, b.sitedims)
    contraction = TCI.Contraction(a_tt, b_tt)
    projector = Projector(
        [[x[1], y[2]] for (x, y) in zip(a.projector, b.projector)], sitedims_ab
    )
    return LazyMatrixMul{T}(coeff, contraction, projector, sitedims_ab, a, b)
end

function LazyMatrixMul(a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(1)) where {T}
    return LazyMatrixMul{T}(a, b; coeff=coeff)
end

function project(
    obj::LazyMatrixMul{T}, prj::Projector; kwargs...
)::LazyMatrixMul{T} where {T}
    projector_a_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(prj, obj.a.projector.sitedims)], obj.a.sitedims
    )
    projector_b_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(obj.b.projector.sitedims, prj)], obj.b.sitedims
    )
    obj.a = project(obj.a, projector_a_new; kwargs...)
    obj.b = project(obj.b, projector_b_new; kwargs...)
    # TO BE FIXED: Cache is thrown away
    return LazyMatrixMul{T}(obj.a, obj.bl; coeff=obj.coeff)
end

function lazymatmul(a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(T)) where {T}
    return LazyMatrixMul{T}(a, b; coeff=coeff)
end

function LazyMatrixMul(a::ProjTensorTrain, b::ProjTensorTrain)
    return LazyMatrixMul(TCI.Contraction(a, b))
end

Base.length(obj::LazyMatrixMul) = length(obj.contraction)

# multi-site-index evaluation
function (obj::LazyMatrixMul{T})(indexset::MMultiIndex)::T where {T}
    return obj.contraction(lineari(obj.sitedims, indexset))
end

# multi-site-index evaluation
function batchevaluateprj(
    obj::LazyMatrixMul{T},
    leftindexset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    NL = length(leftindexset[1])
    NR = length(rightmmultiidxset[1])
    L = length(obj)
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightmmultiidxset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset]
    projector = Int[
        isprojectedat(obj.projector, n) ? _lineari(obj.sitedims[n], obj.projector[n]) : 0
        for n in (NL + 1):(L - NR)
    ]
    returnshape = [
        isprojectedat(obj.projector, n) ? 1 : prod(obj.sitedims[n]) for
        n in (NL + 1):(L - NR)
    ]
    res = TCI.batchevaluate(
        obj.contraction, leftindexset_, rightmmultiidxset_, Val(M), projector
    )
    return reshape(res, length(leftindexset), returnshape..., length(rightmmultiidxset))
end
