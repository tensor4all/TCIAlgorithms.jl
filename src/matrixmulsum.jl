"""
Sum of matrix products of two tensor trains
Two site indices on each site.
"""
struct MatrixProductSum{T} <: TCI.BatchEvaluator{T}
    products::Vector{MatrixProduct{T}}
end

Base.length(obj::MatrixProductSum) = length(obj.products[1])

function Base.lastindex(obj::MatrixProductSum{T}) where {T}
    return lastindex(obj.products[1])
end

function Base.getindex(obj::MatrixProductSum{T}, i) where {T}
    return getindex(obj.products[1], i)
end


function Base.show(io::IO, obj::MatrixProductSum{T}) where {T}
    print(io, "$(typeof(obj)) consisting of $(length(obj.products)) matrix products")
end


function evaluate(obj::MatrixProductSum{T}, indexset::AbstractVector{Int})::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end
    return sum([evaluate(p, indexset) for p in obj.products])
end


function (obj::MatrixProductSum{T})(indexset::AbstractVector{Int})::T where {T}
    return evaluate(obj, indexset)
end


function (obj::MatrixProductSum{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    return sum(
        p(leftindexset, rightindexset, Val(M)) for p in obj.products
    )
end
