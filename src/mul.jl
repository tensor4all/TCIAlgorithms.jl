"""
Lazy evaluation for matrix multiplication of two TTOs
Two site indices on each site.
"""
struct MatrixProduct{T} <: ProjectableEvaluator{T}
    coeff::T
    contraction::TCI.Contraction{T}
    projector::Projector
    sitedims::Vector{Vector{Int}}
    a::ProjTensorTrain{T}
    b::ProjTensorTrain{T}
end

function MatrixProduct{T}(a::ProjTensorTrain, b::ProjTensorTrain; coeff::T=1) where {T}
    # This restriction is due to simulicity and to be removed.
    length.(a.sitedims) .== 2 || error("The number of site indices must be 2")
    length.(b.sitedims) .== 2 || error("The number of site indices must be 2")
    a_tt = TensorTrain{T,4}(a.data, sitedims)
    b_tt = TensorTrain{T,4}(b.data, sitedims)
    contraction = TCI.Contraction(a_tt, b_tt)
    sitedims = [[x, y] for (x, y) in zip(a.sitedims, b.sitedims)]
    projector = Projector(
        [[x[1], y[2]] for (x, y) in zip(ptt1.projector, ptt2.projector)], sitedims
    )
    return MatrixProduct{T}(coeff, contraction, projector, sitedims, a, b)
end

function project(
    obj::MatrixProduct{T},
    prj::Projector;
    kwargs...
)::MatrixProduct{T} where {T}
    projector_a_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(prj, obj.projector_a.sitedims)],
        obj.a.sitedims
    )
    projector_b_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(obj.projector_b.sitedims, prj)],
        obj.b.sitedims
    )
    obj.a = project(obj.a, projector_a_new; kawargs...)
    obj.b = project(obj.b, projector_b_new; kawargs...)
    # TO BE FIXED: Cache is thrown away
    return MatrixProduct{T}(obj.a, obj.bl; coeff=obj.coeff)
end

function mul(a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff::T=1) where {T}
    return MatrixProduct(coeff, TCI.Contraction(a, b))
end

function MatrixProduct(a::ProjTensorTrain, b::ProjTensorTrain)
    return MatrixProduct(TCI.Contraction(a, b))
end

Base.length(obj::MatrixProduct) = length(obj.contraction)

#function Base.lastindex(obj::MatrixProduct{T}) where {T}
#return lastindex(obj.mpo[1])
#end
#
#function Base.getindex(obj::MatrixProduct{T}, i) where {T}
#return getindex(obj.mpo[1], i)
#end
#
#function evaluate(
#obj::MatrixProduct{T}, indexset::AbstractVector{Tuple{Int,Int}}
#)::T where {T}
#return obj.contraction(indexset)
#end

# multi-site-index evaluation
function (obj::MatrixProduct{T})(indexset::MMultiIndex)::T where {T}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

# multi-site-index evaluation
function batchevaluateprj(
    obj::MatrixProduct{T},
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
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]
    projector = Int[
        isprojectedat(obj.projector, n) ? _lineari(obj.sitedims[n], obj.projector[n]) : 0
        for n in (NL + 1):(L - NR)
    ]
    returnshape = [
        isprojectedat(obj.projector, n) ? 1 : prod(obj.sitedims[n]) for
        n in (NL + 1):(L - NR)
    ]
    res = TCI.batchevaluate(obj.contraction, leftindexset_, rightindexset_, Val(M), projector)
    return reshape(res, length(leftindexset), returnshape..., length(rightindexset))
end