struct ProjContainer{T,V<:ProjectableEvaluator{T}} <: ProjectableEvaluator{T}
    data::Vector{V} # Projectors of `data` can overlap with each other
    sitedims::Vector{Vector{Int}}
    projector::Projector # The projector of the container, which is the union of the projectors of `data`


    function ProjContainer{T,V}(data) where {T,V}
        data = V[x for x in data]
        sitedims = data[1].sitedims
        for x in data
            sitedims == x.sitedims || error("Sitedims mismatch")
        end
        projector = reduce(|, x.projector for x in data)
        return new{T,V}(data, sitedims, projector)
    end
end

function Base.iterate(obj::ProjContainer, state=1)
    if state > length(obj.data)
        return nothing
    end
    return (obj.data[state], state + 1)
end

Base.length(obj::ProjContainer) = length(obj.data)
Base.getindex(obj::ProjContainer, index::Int) = obj.data[index]

const ProjTTContainer{T} = ProjContainer{T,ProjTensorTrain{T}}

function ProjTTContainer(data::AbstractVector{ProjTensorTrain{T}}) where {T}
    return ProjContainer{T,ProjTensorTrain{T}}(data)
end

function ProjTTContainer(data::AbstractSet{ProjTensorTrain{T}}) where {T}
    return ProjContainer{T,ProjTensorTrain{T}}(collect(data))
end

#function ProjTTContainer(data) where {T}
    #return ProjContainer{T,ProjTensorTrain{T}}(data)
#end

function Base.reshape(
    obj::ProjContainer{T,V}, sitedims::AbstractVector{<:AbstractVector{Int}}
)::ProjContainer{T,V} where {T,V}
    ProjContainer{T,V}([reshape(x, sitedims) for x in obj.data])
end

function (obj::ProjContainer{T,V})(mmultiidx::MMultiIndex)::T where {T,V}
    return Base.sum(o(mmultiidx) for o in obj.data)
end

function batchevaluateprj(
    obj::ProjContainer{T,V},
    leftmmultiidxset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,V,M}
    M >= 0 || error("The order of the result must be non-negative")
    return Base.sum(batchevaluateprj(o, leftmmultiidxset, rightmmultiidxset, Val(M)) for o in obj.data)
end

#function (obj::ProjContainer{T,V})(
    #leftmmultiidxset::AbstractVector{MMultiIndex},
    #rightmmultiidxset::AbstractVector{MMultiIndex},
    #::Val{M},
#)::Array{T,M + 2} where {T,V,M}
    #return sum(o(leftmmultiidxset, rightmmultiidxset, Val(M)) for o in obj.data)
#end

function Base.show(io::IO, obj::ProjContainer{T,V}) where {T,V}
    return print(io, "ProjContainer{$T,$V} with $(length(obj.data)) elements")
end

function Base.show(io::IO, obj::ProjContainer{T,ProjTensorTrain{T}}) where {T}
    return print(io, "ProjTTContainer{$T} with $(length(obj.data)) elements")
end
