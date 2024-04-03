
function allequal(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all(collection .== c)
end

function _multii(sitedims::Vector{Int}, i::Int)::Vector{Int}
    i <= prod(sitedims) || error("Index out of range $i, $sitedims")
    return if i == 0
        fill(0, length(sitedims))
    else
        collect(Tuple(CartesianIndices(Tuple(sitedims))[i]))
    end
end

function multii(sitedims::Vector{Vector{Int}}, indexset::MultiIndex)::Vector{Vector{Int}}
    return [_multii(sitedims[l], i) for (l, i) in enumerate(indexset)]
end

function _lineari(dims, mi)::Integer
    ci = CartesianIndex(Tuple(mi))
    li = LinearIndices(Tuple(dims))
    return li[ci]
end

function lineari(sitedims::Vector{Vector{Int}}, indexset::Vector{MultiIndex})::Vector{Int}
    return [_lineari(sitedims[l], indexset[l]) for l in 1:length(indexset)]
end

function typesafe_iterators_product(::Val{N}, dims) where {N}
    return Iterators.product(ntuple(i -> 1:dims[i], N)...)
end
