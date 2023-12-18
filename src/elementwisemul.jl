"""
Elementwise product of two tensor trains
One site index on each site.
"""
struct ElementwiseProduct{T} <: TCI.BatchEvaluator{T}
    cache::Vector{TTCache{T}}
    f::Union{Nothing,Function}
end


function ElementwiseProduct(
    tt::Vector{TensorTrain{T,3}};
    f::Union{Nothing,Function} = nothing,
) where {T}
    if length(unique(length.(tt))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    return ElementwiseProduct(TTCache.(tt), f)
end


function evaluate(
    obj::ElementwiseProduct{T},
    indexset::AbstractVector{Int};
    usecache::Bool = true,
)::T where {T}
    if obj.f === nothing
        return prod(.*, evaluate.(obj.tt, indexset; usecache = usecache))
    else
        return obj.f(prod(.*, evaluate.(obj.tt, indexset; usecache = usecache)))
    end
end


function (obj::ElementwiseProduct{T})(indexset::AbstractVector{Int})::T where {T}
    if obj.f === nothing
        return prod(.*, (t(indexset) for t in obj.cache))
    else
        return obj.f(prod(.*, (t(indexset) for t in obj.cache)))
    end
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
    if obj.f === nothing
        return res
    else
        return obj.f.(res)
    end
end

function elementwiseproduct(
    tts::TensorTrain{T,3}...;
    tolerance = 1e-12,
    maxbonddim = typemax(Int),
    f::Union{Nothing,Function} = nothing,
) where {T}
    if !allequal(length.(tts))
        throw(ArgumentError("Cannot multiply TTs with different length: $(length.(tts))"))
    end
    if !all(allequal(TCI.sitedim(tt, i)[1] for tt in tts) for i = 1:length(tts[1]))
        throw(ArgumentError("Cannot multiply TTs with different local dimensions."))
    end
    return TCI.crossinterpolate2(
        T,
        ElementwiseProduct(collect(tts); f = f),
        [d[1] for d in TCI.sitedims(tts[1])];
        tolerance = tolerance,
        maxbonddim = maxbonddim,
    )
end
