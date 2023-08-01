"""
Elementwise product of two tensor trains
One site index on each site.
"""
struct ElementwiseProduct{T} <: TCI.BatchEvaluator{T}
    tt::Vector{TensorTrain{T,3}}
    cache::Vector{TTCache{T}}
end


function ElementwiseProduct(tt::Vector{TensorTrain{T,3}}) where {T}
    if length(unique(length.(tt))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    return ElementwiseProduct(tt, TTCache.(tt))
end


function evaluate(
    obj::ElementwiseProduct{T},
    indexset::AbstractVector{Int};
    usecache::Bool=true)::T where {T}
    return prod(.*, evaluate.(obj.tt, indexset; usecache=usecache))
end


function (obj::ElementwiseProduct{T})(indexset::AbstractVector{Int})::T where {T}
    return prod(.*, (t(indexset) for t in obj.tt))
end


function TCI.batchevaluate(obj::ElementwiseProduct{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{T,M + 2} where {T,M}

    res = TCI.batchevaluate(obj.cache[1], leftindexset, rightindexset, Val(M))
    for c in obj.cache[2:end]
        res = res .* TCI.batchevaluate(c, leftindexset, rightindexset, Val(M))
    end
    return res
end