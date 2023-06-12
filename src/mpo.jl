
"""
    struct MPO{ValueType, N}

Represents a Matrix Product Operator / Tensor Train with N legs on each tensor.
"""
struct MPO{ValueType,N} <: TCI.AbstractTensorTrain{ValueType}
    T::Vector{Array{ValueType,N}}
end

function MPO(TT::TCI.AbstractTensorTrain{ValueType}) where {ValueType}
    new{ValueType}(TT.T)
end

function contractmpotensor(
    f::AbstractArray, fleg::Int, g::AbstractArray, gleg::Int
)
    flegsinv = setdiff(2:ndims(f)-1, fleg + 1)
    glegsinv = setdiff(2:ndims(g)-1, gleg + 1)

    resultndims = ndims(f) + ndims(g) - 2

    resultleftlinkindices = (1, ndims(f))
    resultrightlinkindices = (ndims(f) - 1, resultndims)
    resultfindices = 2:ndims(f)-2
    resultgindices = (2:ndims(g)-2) .+ (ndims(f) - 1)

    permutation = [
        resultleftlinkindices...,
        resultfindices..., resultgindices...,
        resultrightlinkindices...]

    resultleftlinksize = size(f, 1) * size(g, 1)
    resultrightlinksize = size(f, ndims(f)) * size(g, ndims(g))
    resultfsizes = size(f)[flegsinv]
    resultgsizes = size(g)[glegsinv]

    sizes = (resultleftlinksize, resultfsizes..., resultgsizes..., resultrightlinksize)

    return reshape(permutedims(contract(f, fleg + 1, g, gleg + 1), permutation), sizes)
end

"""
    function contract(f::MPO, flegs::Vector{Int}, g::MPO, glegs::Vector{Int})

MPO-MPO contraction between MPO f and g, on legs specified by flegs and glegs.
"""
function contract(f::MPO, flegs::Vector{Int}, g::MPO, glegs::Vector{Int})
    if length(f) != length(g)
        throw(DimensionMismatch("Only MPO of equal length can be contracted."))
    end
    if length(f) != length(flegs)
        throw(DimensionMismatch("Argument flegs must specify legs to contract for each tensor of MPO f."))
    end
    if length(g) != length(glegs)
        throw(DimensionMismatch("Argument flegs must specify legs to contract for each tensor of MPO f."))
    end

    return MPO([
        contractmpotensor(fT, fl, gT, gl)
        for (fT, fl, gT, gl) in zip(f, flegs, g, glegs)])
end

function fusephysicallegs(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

function splitphysicallegs(t::AbstractArray{T}, legdims::Union{AbstractVector{Int}, Tuple}) where {T}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

function recompress(f::MPO{T}; maxbonddim=200, tolerance=1e-8) where {T}
    ffused = MPO([fusephysicallegs(t)[1] for t in f])
    tt, ranks, errors = TCI.crossinterpolate2(
        T,
        ffused,
        [size(t, 2) for t in ffused];
        maxbonddim=maxbonddim,
        tolerance=tolerance
    )
    result = MPO([splitphysicallegs(t, size(ft)[2:end-1]) for (t, ft) in zip(tt, f)])
    return result, ranks, errors
end
