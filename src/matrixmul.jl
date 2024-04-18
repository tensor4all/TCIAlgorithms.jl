"""
Matrix product of two tensor trains
Two site indices on each site.
"""
struct MatrixProduct{T} <: TCI.BatchEvaluator{T}
    mpo::NTuple{2,TensorTrain{T,4}}
    leftcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    rightcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    #a_MPO::MPO
    #b_MPO::MPO
    #sites1::Vector{Index{Int}}
    #sites2::Vector{Index{Int}}
    #sites3::Vector{Index{Int}}
    #links_a::Vector{Index{Int}}
    #links_b::Vector{Index{Int}}
    f::Union{Nothing,Function}
    sitedims::Vector{Vector{Int}}
end

Base.length(obj::MatrixProduct) = length(obj.mpo[1])

function Base.lastindex(obj::MatrixProduct{T}) where {T}
    return lastindex(obj.mpo[1])
end

function Base.getindex(obj::MatrixProduct{T}, i) where {T}
    return getindex(obj.mpo[1], i)
end

function Base.show(io::IO, obj::MatrixProduct{T}) where {T}
    return print(
        io,
        "$(typeof(obj)) of tensor trains with ranks $(TCI.rank(obj.mpo[1])) and $(TCI.rank(obj.mpo[2]))",
    )
end

function MatrixProduct(
    a::TensorTrain{T,4}, b::TensorTrain{T,4}; f::Union{Nothing,Function}=nothing
) where {T}
    mpo = a, b
    if length(unique(length.(mpo))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    for n in 1:length(mpo[1])
        if size(mpo[1][n], 3) != size(mpo[2][n], 2)
            throw(ArgumentError("Tensor trains must share the identical index at n=$n!"))
        end
    end

    localdims1 = [size(mpo[1][n], 2) for n = 1:length(mpo[1])]
    localdims3 = [size(mpo[2][n], 3) for n = 1:length(mpo[2])]

    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims3)]

    return MatrixProduct(
        mpo,
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        f,
        sitedims,
    )
end

_localdims(obj::TensorTrain{<:Any,4}, n::Int)::Tuple{Int,Int} =
    (size(obj[n], 2), size(obj[n], 3))
_localdims(obj::MatrixProduct{<:Any}, n::Int)::Tuple{Int,Int} =
    (size(obj.mpo[1][n], 2), size(obj.mpo[2][n], 3))

function _unfuse_idx(obj::MatrixProduct{T}, n::Int, idx::Int)::Tuple{Int,Int} where {T}
    return reverse(divrem(idx - 1, _localdims(obj, n)[1]) .+ 1)
end

function _fuse_idx(obj::MatrixProduct{T}, n::Int, idx::Tuple{Int,Int})::Int where {T}
    return idx[1] + _localdims(obj, n)[1] * (idx[2] - 1)
end

function _extend_cache(oldcache::Matrix{T}, a_ell::Array{T,4}, b_ell::Array{T,4}, i::Int, j::Int) where {T}
    # (link_a, link_b) * (link_a, s, link_a') => (link_b, s, link_a')
    tmp1 = _contract(oldcache, a_ell[:, i, :, :], (1,), (1,))

    # (link_b, s, link_a') * (link_b, s, link_b') => (link_a', link_b')
    return _contract(tmp1, b_ell[:, :, j, :], (1, 2), (1, 2))
end

# Compute left environment
function evaluateleft(
    obj::MatrixProduct{T}, indexset::AbstractVector{Tuple{Int,Int}}
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    if length(indexset) == 0
        return ones(T, 1, 1)
    end

    ell = length(indexset)
    if ell == 1
        i, j = indexset[1]
        return transpose(a[1][1, i, :, :]) * b[1][1, :, j, :] 
    end

    key = collect(indexset)
    if !(key in keys(obj.leftcache))
        i, j = indexset[end]
        obj.leftcache[key] = _extend_cache(evaluateleft(obj, indexset[1:ell-1]), a[ell], b[ell], i, j)
    end

    return obj.leftcache[key]
end

# Compute right environment
function evaluateright(
    obj::MatrixProduct{T}, indexset::AbstractVector{Tuple{Int,Int}}
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    a, b = obj.mpo

    N = length(obj)

    if length(indexset) == 0
        return ones(T, 1, 1)
    elseif length(indexset) == 1
        i, j = indexset[1]
        return a[end][:, i, :, 1] * transpose(b[end][:, :, j, 1])
    end

    ell = N - length(indexset) + 1

    key = collect(indexset)
    if !(key in keys(obj.rightcache))
        i, j = indexset[1]
        obj.rightcache[key] = _extend_cache(
            evaluateright(obj, indexset[2:end]),
            permutedims(a[ell], (4, 2, 3, 1)),
            permutedims(b[ell], (4, 2, 3, 1)),
            i, j)
    end

    return obj.rightcache[key]
end

function evaluate(obj::MatrixProduct{T}, indexset::AbstractVector{Int})::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    indexset_unfused = [_unfuse_idx(obj, n, indexset[n]) for n in 1:length(obj)]
    return evaluate(obj, indexset_unfused)
end

function evaluate(
    obj::MatrixProduct{T}, indexset::AbstractVector{Tuple{Int,Int}}
)::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    midpoint = div(length(obj), 2)
    res = Base.sum(
        evaluateleft(obj, indexset[1:midpoint]) .*
        evaluateright(obj, indexset[(midpoint + 1):end]),
    )

    if obj.f isa Function
        return obj.f(res)
    else
        return res
    end
end

function (obj::MatrixProduct{T})(indexset::AbstractVector{Int})::T where {T}
    return evaluate(obj, indexset)
end

function (obj::MatrixProduct{T})(
    indexset::AbstractVector{<:AbstractVector{Int}}
)::T where {T}
    return evaluate(obj, lineari(obj.sitedims, indexset))
end

function (obj::MatrixProduct{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    N = length(obj)
    Nr = length(rightindexset[1])
    s_ = length(leftindexset[1]) + 1
    e_ = N -length(rightindexset[1])
    a, b = obj.mpo

    # Unfused index
    leftindexset_unfused = [
        [_unfuse_idx(obj, n, idx) for (n, idx) in enumerate(idxs)] for idxs in leftindexset
    ]
    rightindexset_unfused = [
        [_unfuse_idx(obj, N - Nr + n, idx) for (n, idx) in enumerate(idxs)] for
        idxs in rightindexset
    ]

    t1 = time_ns()
    linkdims_a = vcat(1, TCI.linkdims(a), 1)
    linkdims_b = vcat(1, TCI.linkdims(b), 1)

    left_ = Array{T,3}(undef, length(leftindexset), linkdims_a[s_], linkdims_b[s_])
    for (i, idx) in enumerate(leftindexset_unfused)
        left_[i, :, :] .= evaluateleft(obj, idx)
    end
    t2 = time_ns()

    right_ = Array{T,3}(
        undef,
        linkdims_a[e_+1],
        linkdims_b[e_+1],
        length(rightindexset),
    )
    for (i, idx) in enumerate(rightindexset_unfused)
        right_[:, :, i] .= evaluateright(obj, idx)
    end
    t3 = time_ns()

    # (left_index, link_a, link_b, site[s_] * site'[s_] *  ... * site[e_] * site'[e_])
    leftobj::Array{T,4} = reshape(left_, size(left_)..., 1)
    for n = s_:e_
        #(left_index, link_a, link_b, S) * (link_a, site[n], shared, link_a')
        #  => (left_index, link_b, S, site[n], shared, link_a')
        tmp1 = _contract(leftobj, a[n], (2,), (1,))

        # (left_index, link_b, S, site[n], shared, link_a') * (link_b, shared, site'[n], link_b')
        #  => (left_index, S, site[n], link_a', site'[n], link_b')
        tmp2 = _contract(tmp1, b[n], (2, 5), (1, 2))

        # (left_index, S, site[n], link_a', site'[n], link_b')
        #  => (left_index, link_a', link_b', S, site[n], site'[n]) 
        tmp3 = permutedims(tmp2, (1, 4, 6, 2, 3, 5))

        leftobj = reshape(tmp3, size(tmp3)[1:3]..., :)
    end

    return_size = (
        length(leftindexset),
        ntuple(i->prod(obj.sitedims[i+s_-1]), M)..., 
        length(rightindexset),
    )
    t5 = time_ns()

    # (left_index, link_a, link_b, S) * (link_a, link_b, right_index)
    #   => (left_index, S, right_index)
    res = _contract(leftobj, right_, (2, 3), (1, 2))

    if obj.f isa Function
        res .= obj.f.(res)
    end

    return reshape(res, return_size)
end

function naivecontract(a::TensorTrain{T,4}, b::TensorTrain{T,4})::TensorTrain{T,4} where {T}
    return naivecontract(MatrixProduct(a, b))
end

function naivecontract(obj::MatrixProduct{T})::TensorTrain{T,4} where {T}
    if obj.f isa Function
        error("Cannot contract matrix product with a function.")
    end

    a, b = obj.mpo

    linkdims_a = vcat(1, TCI.linkdims(a), 1)
    linkdims_b = vcat(1, TCI.linkdims(b), 1)
    linkdims_ab = linkdims_a .* linkdims_b

    # (link_a, s1, s2, link_a') * (link_b, s2, s3, link_b')
    #  => (link_a, s1, link_a', link_b, s3, link_b')
    #  => (link_a, link_b, s1, s3, link_a', link_b')
    sitetensors = [reshape(permutedims(_contract(obj.mpo[1][n], obj.mpo[2][n], (3,), (2,)), (1, 4, 2, 5, 3, 6)), linkdims_ab[n], obj.sitedims[n]..., linkdims_ab[n+1]) for n = 1:length(obj)]

    return TensorTrain{T,4}(sitetensors)
end

function _reshape_fusesites(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:(end - 1)]), shape[end]), shape[2:(end - 1)]
end

function _reshape_splitsites(
    t::AbstractArray{T}, legdims::Union{AbstractVector{Int},Tuple}
) where {T}
    return reshape(t, size(t, 1), legdims..., size(t, ndims(t)))
end

function contract_TCI(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    initialpivots::Union{Int,Vector{MultiIndex}}=10,
    f::Union{Nothing,Function} = nothing,
    kwargs...
) where {ValueType}
    if length(A) != length(B)
        throw(ArgumentError("Cannot contract tensor trains with different length."))
    end
    if !all([TCI.sitedim(A, i)[2] == TCI.sitedim(B, i)[1] for i in 1:length(A)])
        throw(
            ArgumentError(
                "Cannot contract tensor trains with non-matching site dimensions."
            ),
        )
    end
    matrixproduct = MatrixProduct(A, B; f = f)
    localdims = prod.(matrixproduct.sitedims)
    if initialpivots isa Int
        initialpivots = findinitialpivots(matrixproduct, localdims, initialpivots)
        if isempty(initialpivots)
            error("No initial pivots found.")
        end
    end

    tci, ranks, errors = TCI.crossinterpolate2(
        ValueType,
        matrixproduct,
        localdims,
        initialpivots;
        kwargs...,
    )
    legdims = [_localdims(matrixproduct, i) for i = 1:length(tci)]
    return TCI.TensorTrain{ValueType,4}(
        [_reshape_splitsites(t, d) for (t,d) in zip(tci, legdims)]
    )
    legdims = [_localdims(matrixproduct, i) for i in 1:length(tci)]
    return TCI.TensorTrain{ValueType,4}([
        _reshape_splitsites(t, d) for (t, d) in zip(tci, legdims)
    ])
end

function contract(
    A::TensorTrain{ValueType,4},
    B::TensorTrain{ValueType,4};
    algorithm = "TCI",
    tolerance::Float64 = 1e-12,
    maxbonddim::Int = typemax(Int),
    f::Union{Nothing,Function} = nothing,
    kwargs...
) where {ValueType}
    if algorithm == "TCI"
        return contract_TCI(A, B; tolerance = tolerance, maxbonddim = maxbonddim, f = f, kwargs...)
    elseif algorithm in ["density matrix", "fit"]
        throw(ArgumentError("Algorithm $algorithm is not implemented yet"))
    else
        throw(ArgumentError("Unknown algorithm $algorithm."))
    end
end
