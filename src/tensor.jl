function reshapetomatrix(A::AbstractArray, firstlegs::Union{AbstractVector{Int},Tuple})
    lastlegs = setdiff(1:ndims(A), firstlegs)
    permutation = vcat(firstlegs, lastlegs)
    Asizes = (size(A)[firstlegs], size(A)[lastlegs])
    matrixdimensions = (prod(Asizes[1]), prod(Asizes[2]))
    return reshape(permutedims(A, permutation), matrixdimensions), Asizes[2]
end

function contract(
    A::AbstractArray,
    Alegs::Union{AbstractVector{Int},Tuple},
    B::AbstractArray,
    Blegs::Union{AbstractVector{Int},Tuple},
)
    Amatrix, Alegdims = reshapetomatrix(A, Alegs)
    Bmatrix, Blegdims = reshapetomatrix(B, Blegs)
    result = transpose(Amatrix) * Bmatrix
    return reshape(result, Alegdims..., Blegdims...)
end

function contract(A::AbstractArray, Alegs::Int, B::AbstractArray, Blegs::Int)
    return contract(A, [Alegs], B, [Blegs])
end

"""
Elementwise product in one index:
``C_{ijk} = A_{ij} B_{jk}``
"""
function deltaproduct(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    if size(A, 2) != size(B, 1)
        throw(
            DimensionMismatch(
                "Second dimension of A must have same length as first dimension of B.",
            ),
        )
    end
    C = Array{T,3}(undef, size(A, 1), size(A, 2), size(B, 2))
    for j in axes(A, 2)
        C[:, j, :] = A[:, j] * transpose(B[j, :])
    end
    return C
end

function deltaproduct(
    A::AbstractArray,
    Alegs::Union{AbstractVector{Int},Tuple},
    B::AbstractArray,
    Blegs::Union{AbstractVector{Int},Tuple},
)
    Amatrix, Alegdims = reshapetomatrix(A, Alegs)
    Bmatrix, Blegdims = reshapetomatrix(B, Blegs)
    result = deltaproduct(transpose(Amatrix), Bmatrix)
    return reshape(result, Alegdims..., size(A)[Alegs]..., Blegdims...)
end
