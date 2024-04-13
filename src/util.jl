
function allequal(collection)
    if isempty(collection)
        return true
    end
    c = first(collection)
    return all(collection .== c)
end



function _multii(sitedims::Vector{Int}, i::Int)::Vector{Int}
    i <= prod(sitedims) || error("Index out of range $i, $sitedims")
    i == 0 ? fill(0, length(sitedims)) : collect(Tuple(CartesianIndices(Tuple(sitedims))[i]))
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
    Iterators.product(ntuple(i->1:dims[i], N)...)
end


function findinitialpivots(f, localdims, nmaxpivots)::Vector{MultiIndex}
    pivots = MultiIndex[]
    for _ in 1:nmaxpivots
        pivot_ = [rand(1:d) for d in localdims]
        pivot_ = TCI.optfirstpivot(f, localdims, pivot_)
        if abs(f(pivot_)) == 0.0
            continue
        end
        push!(pivots, pivot_)
    end
    return pivots
end


_getindex(x, indices) = ntuple(i->x[indices[i]], length(indices))

function _contract(
    a::AbstractArray{T1,N1},
    b::AbstractArray{T2,N2},
    idx_a::NTuple{n1,Int},
    idx_b::NTuple{n2,Int}
    ) where {T1, T2, N1, N2, n1, n2}
    length(idx_a) == length(idx_b) || error("length(idx_a) != length(idx_b)")
    # check if idx_a contains only unique elements
    length(unique(idx_a)) == length(idx_a) || error("idx_a contains duplicate elements")
    # check if idx_b contains only unique elements
    length(unique(idx_b)) == length(idx_b) || error("idx_b contains duplicate elements")
    # check if idx_a and idx_b are subsets of 1:N1 and 1:N2
    all(1 <= idx <= N1 for idx in idx_a) || error("idx_a contains elements out of range")
    all(1 <= idx <= N2 for idx in idx_b) || error("idx_b contains elements out of range")

    rest_idx_a = setdiff(1:N1, idx_a)
    rest_idx_b = setdiff(1:N2, idx_b)

    amat = reshape(permutedims(a, (rest_idx_a..., idx_a...)), prod(_getindex(size(a), rest_idx_a)), prod(_getindex(size(a), idx_a)))
    bmat = reshape(permutedims(b, (idx_b..., rest_idx_b...)), prod(_getindex(size(b), idx_b)), prod(_getindex(size(b), rest_idx_b)))

    return reshape(amat * bmat, _getindex(size(a), rest_idx_a)..., _getindex(size(b), rest_idx_b)...)
end