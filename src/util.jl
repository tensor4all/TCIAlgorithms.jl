
function allequal(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all(collection .== c)
end

function multii(sitedims::Vector{Vector{Int}}, indexset::MultiIndex)::Vector{Vector{Int}}
    return [
        collect(Tuple(CartesianIndices(Tuple(sitedims[l]))[i]))
        for (l, i) in enumerate(indexset)]
end

function _lineari(dims, mi)::Integer
    ci = CartesianIndex(Tuple(mi))
    li = LinearIndices(dims)
    return li[ci]
end

function lineari(sitedims::Vector{Vector{Int}}, indexset::Vector{MultiIndex})::Vector{Int}
    return [_lineari(sitedims[l], indexset[l]) for l in 1:length(indexset)]
end
