"""
Elementwise product of two tensor trains
One site index on each site.
"""
struct ElementwiseProduct{T} <: TCI.BatchEvaluator{T}
    cache::Vector{TTCache{T}}
end


function ElementwiseProduct(tt::Vector{TensorTrain{T,3}}) where {T}
    if length(unique(length.(tt))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    return ElementwiseProduct(TTCache.(tt))
end


function evaluate(
    obj::ElementwiseProduct{T},
    indexset::AbstractVector{Int};
    usecache::Bool = true,
)::T where {T}
    return prod(.*, evaluate.(obj.tt, indexset; usecache = usecache))
end


function (obj::ElementwiseProduct{T})(indexset::AbstractVector{Int})::T where {T}
    return prod(.*, (t(indexset) for t in obj.cache))
end


function TCI.batchevaluate(
    obj::ElementwiseProduct{T},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}

    res = TCI.batchevaluate(obj.cache[1], leftindexset, rightindexset, Val(M))
    for c in obj.cache[2:end]
        res .*= TCI.batchevaluate(c, leftindexset, rightindexset, Val(M))
    end
    return res
end

function elementwiseproduct(
    tts::TensorTrain{T, 3}...;
    tolerance=1e-12,
    maxbonddim=typemax(Int)
) where {T}
    if !allequal(length.(tts))
        throw(ArgumentError("Cannot multiply TTs with different length: $(length.(tts))"))
    end
    if !all(allequal(TCI.sitedim(tt, i)[1] for tt in tts) for i in 1:length(tts[1]))
        throw(ArgumentError("Cannot multiply TTs with different local dimensions."))
    end
    return TCI.crossinterpolate2(
        T,
        ElementwiseProduct(collect(tts)),
        [d[1] for d in TCI.sitedims(tts[1])];
        tolerance=tolerance,
        maxbonddim=maxbonddim
    )
end
