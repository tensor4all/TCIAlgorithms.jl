"""
Simple Tensor Train implementation in Julia.
Arbitrary number of site indices are supported.
"""
mutable struct DTensorTrain{T} <: ProjectableEvaluator{T}
    sitetensors::Vector{Array{T}}
    sitedims::Vector{Vector{Int}}
end

function DTensorTrain(sitetensors::AbstractVector{Array{T,N}}) where {T,N}
    sitedims = [collect(size(t)[2:(end - 1)]) for t in sitetensors]
    return DTensorTrain{T}(
        sitetensors,
        sitedims;
        sitetensors_fused=[
            reshape(x, size(x, 1), :, size(x)[end]) for x in obj.sitetensors
        ],
    )
end

# multi-site-index evaluation
function (obj::DTensorTrain{T})(
    leftindexset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightmmultiidxset[1])
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightmmultiidxset_ = [
        lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset
    ]

    sitetensors_fused = [reshape(x, size(x, 1), :, size(x)[end]) for x in obj.sitetensors]

    return TensorTrain(sitetensors_fused)(leftindexset_, rightmmultiidxset_, Val(M))
end
