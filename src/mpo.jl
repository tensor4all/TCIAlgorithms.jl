
"""
    struct MPO{ValueType, N}

Represents a Matrix Product Operator / Tensor Train with N legs on each tensor.
"""
struct MPO{ValueType} <: TCI.CachedTensorTrain{ValueType}
    T::Vector{Array{ValueType}}
    cacheleft::Vector{Dict{Vector{Vector{Int}},Array{ValueType}}}
    cacheright::Vector{Dict{Vector{Vector{Int}},Array{ValueType}}}

    function MPO(T::AbstractVector{<:AbstractArray{ValueType}}) where {ValueType}
        return new{ValueType}(
            T,
            [Dict{Vector{Vector{Int}},Array{ValueType}}() for _ in T],
            [Dict{Vector{Vector{Int}},Array{ValueType}}() for _ in T],
        )
    end
end

function MPO(TT::TCI.AbstractTensorTrain{ValueType}) where {ValueType}
    return MPO(TT.T)
end

function ttcache(tt::MPO{V}, leftright::Symbol, b::Int) where {V}
    if leftright == :left
        return tt.cacheleft[b]
    elseif leftright == :right
        return tt.cacheright[b]
    else
        throw(ArgumentError("Choose from left or right cache."))
    end
end

function evaluateleft(tt::MPO{V}, indexset::AbstractVector{<:AbstractVector{Int}}) where {V}
    if isempty(indexset) || length(indexset) > length(tt)
        throw(
            ArgumentError(
                "For evaluateleft, number of indices must be smaller or equal to the number of MPO legs.",
            ),
        )
    end

    ell = length(indexset)
    if ell == 1
        return tt[1][:, indexset[1]..., :]
    end

    cache = ttcache(tt, :left, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] =
            evaluateleft(tt, indexset[1:(ell - 1)]) * tt[ell][:, indexset[ell]..., :]
    end
    return cache[key]
end

function evaluateright(
    tt::MPO{V}, indexset::AbstractVector{<:AbstractVector{Int}}
) where {V}
    if isempty(indexset) || length(indexset) > length(tt)
        throw(
            ArgumentError(
                "For evaluateright, number of indices must be smaller or equal to the number of MPO legs.",
            ),
        )
    end

    if length(indexset) == 1
        return tt[end][:, indexset[1]..., :]
    end
    ell = length(tt) - length(indexset) + 1

    cache = ttcache(tt, :right, ell)
    key = collect(indexset)
    if !(key in keys(cache))
        cache[key] = tt[ell][:, indexset[1]..., :] * evaluateright(tt, indexset[2:end])
    end
    return cache[key]
end

function evaluate(
    tt::MPO{V}, indexset::AbstractVector{<:AbstractVector{Int}}; usecache::Bool=true
)::V where {V}
    if length(tt) != length(indexset)
        throw(
            ArgumentError(
                "To evaluate a tensor train of length $(length(tt)), need $(length(tt)) index values, but only got $(length(indexset)).",
            ),
        )
    end
    if usecache
        midpoint = div(length(tt), 2)
        return Base.sum(
            evaluateleft(tt, indexset[1:midpoint]) *
            evaluateright(tt, indexset[(midpoint + 1):end]),
        )
    else
        return Base.sum(prod(T[:, i..., :] for (T, i) in zip(tt, indexset)))
    end
end

function _fusedtensor(tt::MPO{V}, i::Int)::Array{V,3} where {V}
    T_ = tt.T[i]
    return reshape(T_, size(T_)[1], :, size(T_)[end])
end

function batchevaluate(
    tt::MPO{V},
    leftindexset::AbstractVector{<:AbstractVector{<:AbstractVector{Int}}},
    rightindexset::AbstractVector{<:AbstractVector{<:AbstractVector{Int}}},
    ::Val{M},
)::Array{V,M + 2} where {V,M}
    N = length(tt.T)
    nleft = length(leftindexset[1])
    nright = length(rightindexset[1])
    nleftindexset = length(leftindexset)
    nrightindexset = length(rightindexset)
    ncent = N - nleft - nright
    lenv = vcat([evaluateleft(tt, lindex) for (il, lindex) in enumerate(leftindexset)]...) # nleftindexset x D
    renv = hcat([evaluateright(tt, rindex) for (ir, rindex) in enumerate(rightindexset)]...) # D x nrightindexset

    localdim = zeros(Int, ncent)
    for n in (nleft + 1):(N - nright)
        # (nleftindexset, d, ..., d, D) x (D, d, D)
        T_ = _fusedtensor(tt, n)
        localdim[n - nleft] = size(T_, 2)
        bonddim_ = size(T_, 1)
        lenv = reshape(lenv, :, bonddim_) * reshape(T_, bonddim_, :)
    end

    # (nleftindexset, d...d * D) x (D, nrightindexset)
    bonddim_ = size(renv, 1)
    lenv = reshape(lenv, :, bonddim_) * reshape(renv, bonddim_, :)

    return reshape(lenv, nleftindexset, localdim..., nrightindexset)
end

function evaluate(
    tt::MPO{V}, indexset::Union{AbstractVector{Int},NTuple{N,Int}}; usecache::Bool=true
)::V where {N,V}
    return evaluate(tt, [[i] for i in indexset]; usecache=usecache)
end

function fuselinks(t::AbstractArray{T}, nlinks::Int)::Array{T} where {T}
    s = size(t)
    return reshape(
        t,
        prod(s[1:nlinks]),
        s[(nlinks + 1):(end - nlinks)]...,
        prod(s[(end - nlinks + 1):end]),
    )
end

function contractmpotensor(f::AbstractArray, fleg::Int, g::AbstractArray, gleg::Int)
    sf = ndims(f) - 1
    stot = sf + ndims(g) - 1
    return fuselinks(
        permutedims(
            contract(f, fleg + 1, g, gleg + 1),
            (1, sf + 1, 2:(sf - 1)..., (sf + 2):(stot - 1)..., sf, stot),
        ),
        2,
    )
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
        throw(
            DimensionMismatch(
                "Argument flegs must specify legs to contract for each tensor of MPO f."
            ),
        )
    end
    if length(g) != length(glegs)
        throw(
            DimensionMismatch(
                "Argument flegs must specify legs to contract for each tensor of MPO f."
            ),
        )
    end

    return MPO([
        contractmpotensor(fT, fl, gT, gl) for (fT, fl, gT, gl) in zip(f, flegs, g, glegs)
    ])
end

"""
    function contract(f::MPO, flegs::Int, g::MPO, glegs::Int)

MPO-MPO contraction between MPO f and g, on legs specified by flegs and glegs.
"""
function contract(f::MPO, fleg::Int, g::MPO, gleg::Int)
    return contract(f, [fleg], g, [gleg])
end

function fusephysicallegs(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:(end - 1)]), shape[end]), shape[2:(end - 1)]
end

function splitphysicallegs(
    t::AbstractArray{T}, legdims::Union{AbstractVector{Int},Tuple}
) where {T}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

function fitmpo(f::MPO{T}; maxbonddim=200, tolerance=1e-8) where {T}
    ffused = MPO([fusephysicallegs(t)[1] for t in f])
    tt, ranks, errors = TCI.crossinterpolate2(
        T,
        q -> evaluate(ffused, q),
        [size(t, 2) for t in ffused];
        maxbonddim=maxbonddim,
        tolerance=tolerance,
    )
    result = MPO([splitphysicallegs(t, size(ft)[2:(end - 1)]) for (t, ft) in zip(tt, f)])
    return result, ranks, errors
end

function multiplympotensor(
    ft::AbstractArray{T},
    flegs::Union{AbstractVector{Int},Tuple},
    gt::AbstractArray{T},
    glegs::Union{AbstractVector{Int},Tuple},
) where {T}
    ncontract = length(flegs)
    nf = ndims(ft) - ncontract
    ng = ndims(gt) - ncontract
    D = permutedims(
        deltaproduct(ft, flegs .+ 1, gt, glegs .+ 1),
        [
            1,
            nf + ncontract + 1, # Left links
            2:(nf - 1)...,
            (nf .+ (1:ncontract))...,
            (nf + ncontract .+ (2:(ng - 1)))...,
            nf,
            nf + ncontract + ng,
        ],
    )
    return fuselinks(D, 2)
end

"""
Elementwise multiplication in indices flegs, glegs
"""
function multiply(
    f::MPO{T},
    flegs::Union{AbstractVector{Int},Tuple},
    g::MPO{T},
    glegs::Union{AbstractVector{Int},Tuple},
)::MPO{T} where {T}
    return MPO([multiplympotensor(ft, flegs, gt, glegs) for (ft, gt) in zip(f, g)])
end

"""
Elementwise multiplication in indices flegs, glegs
"""
function multiply(f::MPO{T}, fleg::Int, g::MPO{T}, gleg::Int)::MPO{T} where {T}
    return multiply(f, [fleg], g, [gleg])
end
