_sumdropdims(x, dims) = dropdims(sum(x; dims=dims); dims=dims)

function integrate(obj::TensorTrain{T,N})::T where {T,N}
    res::Array{T,3} = prod(_sumdropdims(reshape(t, size(t, 1), :, size(t, N)), 2) for t in obj)
    return only(res)
end