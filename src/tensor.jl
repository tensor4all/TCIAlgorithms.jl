
function contract(
    A::AbstractArray, Alegs::AbstractVector{Int},
    B::AbstractArray, Blegs::AbstractVector{Int}
)
    Alegsinv = setdiff(1:ndims(A), Alegs)
    Aperm = vcat(Alegsinv, Alegs)
    Adims = (prod(size(A)[Alegsinv]), prod(size(A)[Alegs]))

    Blegsinv = setdiff(1:ndims(B), Blegs)
    Bperm = vcat(Blegs, Blegsinv)
    Bdims = (prod(size(B)[Blegs]), prod(size(B)[Blegsinv]))

    result = reshape(permutedims(A, Aperm), Adims) * reshape(permutedims(B, Bperm), Bdims)
    return reshape(result, (size(A)[Alegsinv]..., size(B)[Blegsinv]...))
end

function contract(A::AbstractArray, Alegs::Int, B::AbstractArray, Blegs::Int)
    return contract(A, [Alegs], B, [Blegs])
end
