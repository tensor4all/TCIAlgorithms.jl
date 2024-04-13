"""
Matrix product of two tensor trains
Two site indices on each site.
"""
struct MatrixProduct{T} <: TCI.BatchEvaluator{T}
    mpo::NTuple{2,TensorTrain{T,4}}
    leftcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    rightcache::Dict{Vector{Tuple{Int,Int}},Matrix{T}}
    a_MPO::MPO
    b_MPO::MPO
    sites1::Vector{Index{Int}}
    sites2::Vector{Index{Int}}
    sites3::Vector{Index{Int}}
    links_a::Vector{Index{Int}}
    links_b::Vector{Index{Int}}
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
    print(
        io,
        "$(typeof(obj)) of tensor trains with ranks $(TCI.rank(obj.mpo[1])) and $(TCI.rank(obj.mpo[2]))",
    )
end

function MatrixProduct(
    a::TensorTrain{T,4},
    b::TensorTrain{T,4};
    f::Union{Nothing,Function} = nothing,
) where {T}
    mpo = a, b
    if length(unique(length.(mpo))) > 1
        throw(ArgumentError("Tensor trains must have the same length."))
    end
    for n = 1:length(mpo[1])
        if size(mpo[1][n], 3) != size(mpo[2][n], 2)
            throw(ArgumentError("Tensor trains must share the identical index at n=$n!"))
        end
    end

    N = length(mpo[1])
    localdims1 = [size(mpo[1][n], 2) for n = 1:length(mpo[1])]
    localdims2 = [size(mpo[1][n], 3) for n = 1:length(mpo[1])]
    localdims3 = [size(mpo[2][n], 3) for n = 1:length(mpo[2])]
    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims3)]

    bonddims_a = vcat([size(mpo[1][n], 1) for n = 1:length(mpo[1])], 1)
    bonddims_b = vcat([size(mpo[2][n], 1) for n = 1:length(mpo[2])], 1)

    links_a = [Index(bonddims_a[n], "Link,l=$n") for n = 1:N+1]
    links_b = [Index(bonddims_b[n], "Link,l=$n") for n = 1:N+1]

    sites1 = [Index(localdims1[n], "Site1=$n") for n = 1:N]
    sites2 = [Index(localdims2[n], "Site2=$n") for n = 1:N]
    sites3 = [Index(localdims3[n], "Site3=$n") for n = 1:N]

    a_MPO =
        MPO([ITensor(a[n], links_a[n], sites1[n], sites2[n], links_a[n+1]) for n = 1:N])
    b_MPO =
        MPO([ITensor(b[n], links_b[n], sites2[n], sites3[n], links_b[n+1]) for n = 1:N])

    return MatrixProduct(
        mpo,
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        Dict{Vector{Tuple{Int,Int}},Matrix{T}}(),
        a_MPO,
        b_MPO,
        sites1,
        sites2,
        sites3,
        links_a,
        links_b,
        f,
        sitedims
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

# Compute left environment
function evaluateleft(
    obj::MatrixProduct{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    if length(indexset) == 0
        return ones(T, 1, 1)
    end

    ell = length(indexset)
    if ell == 1
        i, j = indexset[1]
        a_ = obj.a_MPO[1] * onehot(obj.sites1[1] => i) * onehot(obj.links_a[1] => 1)
        b_ = obj.b_MPO[1] * onehot(obj.sites3[1] => j) * onehot(obj.links_b[1] => 1)
        return Array(a_ * b_, [obj.links_a[2], obj.links_b[2]])
    end

    key = collect(indexset)
    if !(key in keys(obj.leftcache))
        # (v1, v2)
        idx_v1 = obj.links_a[ell]
        idx_v2 = obj.links_b[ell]
        left = ITensor(evaluateleft(obj, indexset[1:ell-1]), (idx_v1, idx_v2))

        i, j = indexset[end]

        # (v1, v2) * (v1, k, v1') = (v2, k, v1')
        idx_v1p = obj.links_a[ell+1]
        idx_v2p = obj.links_b[ell+1]

        res_tensor =
            left *
            (obj.a_MPO[ell] * onehot(obj.sites1[ell] => i)) *
            (obj.b_MPO[ell] * onehot(obj.sites3[ell] => j))

        obj.leftcache[key] = Array(res_tensor, [idx_v1p, idx_v2p])
    end

    return obj.leftcache[key]
end



# Compute right environment
function evaluateright(
    obj::MatrixProduct{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::Matrix{T} where {T}
    if length(indexset) >= length(obj.mpo[1])
        error("Invalid indexset: $indexset")
    end

    N = length(obj)

    if length(indexset) == 0
        return ones(T, 1, 1)
    elseif length(indexset) == 1
        i, j = indexset[1]
        a_ = obj.a_MPO[end] * onehot(obj.sites1[end] => i) * onehot(obj.links_a[end] => 1)
        b_ = obj.b_MPO[end] * onehot(obj.sites3[end] => j) * onehot(obj.links_b[end] => 1)
        return Array(a_ * b_, [obj.links_a[N], obj.links_b[N]])
    end

    ell = N - length(indexset) + 1

    key = collect(indexset)
    if !(key in keys(obj.rightcache))
        # (v1, v2)
        idx_v1 = obj.links_a[ell+1]
        idx_v2 = obj.links_b[ell+1]
        right = ITensor(evaluateright(obj, indexset[2:end]), (idx_v1, idx_v2))

        i, j = indexset[1]
        res_tensor =
            right *
            obj.a_MPO[ell] *
            onehot(obj.sites1[ell] => i) *
            obj.b_MPO[ell] *
            onehot(obj.sites3[ell] => j)

        obj.rightcache[key] = Array(res_tensor, [obj.links_a[ell], obj.links_b[ell]])
    end

    return obj.rightcache[key]
end


function evaluate(obj::MatrixProduct{T}, indexset::AbstractVector{Int})::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    indexset_unfused = [_unfuse_idx(obj, n, indexset[n]) for n = 1:length(obj)]
    return evaluate(obj, indexset_unfused)
end

function evaluate(
    obj::MatrixProduct{T},
    indexset::AbstractVector{Tuple{Int,Int}},
)::T where {T}
    if length(obj) != length(indexset)
        error("Length mismatch: $(length(obj)) != $(length(indexset))")
    end

    midpoint = div(length(obj), 2)
    res = sum(
        evaluateleft(obj, indexset[1:midpoint]) .*
        evaluateright(obj, indexset[midpoint+1:end]),
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

function (obj::MatrixProduct{T})(indexset::AbstractVector{<:AbstractVector{Int}})::T where {T}
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
    e_ = N - length(rightindexset[1])

    # Unfused index
    leftindexset_unfused = [
        [_unfuse_idx(obj, n, idx) for (n, idx) in enumerate(idxs)] for idxs in leftindexset
    ]
    rightindexset_unfused = [
        [_unfuse_idx(obj, N - Nr + n, idx) for (n, idx) in enumerate(idxs)] for
        idxs in rightindexset
    ]

    t1 = time_ns()
    left_ =
        Array{T,3}(undef, dim(obj.links_a[s_]), dim(obj.links_b[s_]), length(leftindexset))
    for (i, idx) in enumerate(leftindexset_unfused)
        left_[:, :, i] .= evaluateleft(obj, idx)
    end
    t2 = time_ns()

    right_ = Array{T,3}(
        undef,
        dim(obj.links_a[e_+1]),
        dim(obj.links_b[e_+1]),
        length(rightindexset),
    )
    for (i, idx) in enumerate(rightindexset_unfused)
        right_[:, :, i] .= evaluateright(obj, idx)
    end
    t3 = time_ns()

    index_left = Index(length(leftindexset), "left")
    index_right = Index(length(rightindexset), "right")

    res = ITensor(left_, obj.links_a[s_], obj.links_b[s_], index_left)
    for n = s_:e_
        res *= obj.a_MPO[n]
        res *= obj.b_MPO[n]
    end
    t4 = time_ns()
    res *= ITensor(right_, obj.links_a[e_+1], obj.links_b[e_+1], index_right)

    res_inds = vcat(
        index_left,
        collect(Iterators.flatten(zip(obj.sites1[s_:e_], obj.sites3[s_:e_]))),
        index_right,
    )

    res_size = vcat(
        dim(index_left),
        [dim(s1) * dim(s3) for (s1, s3) in zip(obj.sites1[s_:e_], obj.sites3[s_:e_])],
        dim(index_right),
    )
    t5 = time_ns()

    if obj.f isa Function
        res .= obj.f.(res)
    end
    #println("1: ", (t2 - t1)*1e-9, " sec")
    #println("2: ", (t3 - t2)*1e-9, " sec")
    #println("3: ", (t4 - t3)*1e-9, " sec")
    #println("4: ", (t5 - t4)*1e-9, " sec")

    return reshape(Array(res, res_inds), res_size...)
end


function _contract(obj::MatrixProduct)::MPO
    if obj.f isa Function
        error("Cannot contract matrix product with a function.")
    end
    a_MPO = copy(obj.a_MPO)
    a_MPO[1] *= onehot(obj.links_a[1] => 1)
    a_MPO[end] *= onehot(obj.links_a[end] => 1)

    b_MPO = copy(obj.b_MPO)
    b_MPO[1] *= onehot(obj.links_b[1] => 1)
    b_MPO[end] *= onehot(obj.links_b[end] => 1)

    return ITensors.contract(a_MPO, b_MPO; alg = "naive")
end

function _reshape_fusesites(t::AbstractArray{T}) where {T}
    shape = size(t)
    return reshape(t, shape[1], prod(shape[2:end-1]), shape[end]), shape[2:end-1]
end

function _reshape_splitsites(
    t::AbstractArray{T},
    legdims::Union{AbstractVector{Int},Tuple},
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
    if !all([TCI.sitedim(A, i)[2] == TCI.sitedim(B, i)[1] for i = 1:length(A)])
        throw(
            ArgumentError(
                "Cannot contract tensor trains with non-matching site dimensions.",
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
