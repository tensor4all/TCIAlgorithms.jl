"""
Elementwise product of two tensor trains
One site index on each site.
"""
struct ElementwiseProduct{T} <: TCI.BatchEvaluator{T}
    a::TensorTrain{T,3}
    b::TensorTrain{T,3}
    a_cache::TTCache{T}
    b_cache::TTCache{T}
end


function ElementwiseProduct(a::TensorTrain{T,3}, b::TensorTrain{T,3}) where {T}
    if length(a) != length(b)
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    a_cache = TTCache(a)
    b_cache = TTCache(b)
    return ElementwiseProduct(a, b, a_cache, b_cache)
end


function evaluate(
    obj::ElementwiseProduct{T},
    indexset::AbstractVector{Int};
    usecache::Bool=true)::T where {T}
    return evaluate(obj.a, indexset; usecache=usecache) .* evaluate(obj.b, indexset; usecache=usecache)
end


function (obj::ElementwiseProduct{T})(indexset::AbstractVector{Int})::T where {T}
    return obj.a(indexset) .* obj.b(indexset)
end


function TCI.batchevaluate(obj::ElementwiseProduct{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M})::Array{T,M + 2} where {T,M}

    return TCI.batchevaluate(obj.a_cache, leftindexset, rightindexset, Val(M)) .* TCI.batchevaluate(obj.b_cache, leftindexset, rightindexset, Val(M))
end