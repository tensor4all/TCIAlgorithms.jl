function sum(obj::TensorTrain{T,N})::T where {T,N}
    return TCI.sum(obj)
end
