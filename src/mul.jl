"""
Lazy evaluation for matrix multiplication of two TTOs
Two site indices on each site.
"""
struct MatrixProduct{T} <: ProjectableEvaluator{T}
    contraction::TCI.Contraction{T}
end



# TO BE IMPLEMENTED, projection

function mul(a::ProjTensorTrain, b::ProjTensorTrain)
    return MatrixProduct(TCI.Contraction(a, b))
end

function MatrixProduct(a::ProjTensorTrain, b::ProjTensorTrain)
    return MatrixProduct(TCI.Contraction(a, b))
end

Base.length(obj::MatrixProduct) = length(obj.contraction)

function Base.lastindex(obj::MatrixProduct{T}) where {T}
    return lastindex(obj.mpo[1])
end

function Base.getindex(obj::MatrixProduct{T}, i) where {T}
    return getindex(obj.mpo[1], i)
end

function evaluate(
    obj::MatrixProduct{T}, indexset::AbstractVector{Tuple{Int,Int}}
)::T where {T}
    return obj.contraction(indexset)
end

# multi-site-index evaluation
function (obj::MatrixProduct{T})(indexset::MMultiIndex)::T where {T}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

# multi-site-index evaluation
function (obj::MatrixProduct{T})(
    leftindexset::AbstractVector{MMultiIndex},
    rightindexset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]

    result = obj(leftindexset_, rightindexset_, Val(M))
    return result
end