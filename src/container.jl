struct ProjContainer{T,V<:ProjectableEvaluator{T}} <: ProjectableEvaluator{T}
    data::Vector{V}
    sitedims::Vector{Vector{Int}}
    projector::Projector

    function ProjContainer{T,V}(data::Vector{V}) where {T,V}
        sitedims = data[1].sitedims
        for x in data
            sitedims == x.sitedims || error("Sitedims mismatch")
        end
        projector = reduce(|, x.projector for x in data)
        new{T,V}(data, sitedims, projector)
    end
end

const ProjTTContainer{T} = ProjContainer{T,ProjTensorTrain{T}}

function ProjTTContainer(data::Vector{ProjTensorTrain{T}}) where {T}
    return ProjContainer{T,ProjTensorTrain{T}}(data)
end

function (obj::ProjContainer{T,V})(mmultiidx::MMultiIndex)::T where {T,V}
    return Base.sum(o(mmultiidx) for o in obj.data)
end

function (obj::ProjContainer{T,V})(
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,V,M}
    return sum(o(leftmmultiidxset, rightmmultiidxset, Val(M)) for o in obj.data)
end

Base.show(io::IO, obj::ProjContainer{T,V}) where {T,V} = print(io, "ProjContainer{$T,$V} with $(length(obj.data)) elements")

Base.show(io::IO, obj::ProjContainer{T,ProjTensorTrain{T}}) where {T} = print(io, "ProjTTContainer{$T} with $(length(obj.data)) elements")