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
    return ProjContainer{T,V}([reshape(x, sitedims) for x in obj.data])
end

function approxtt(
    obj::ProjContainer{T,V}; maxbonddim=typemax(Int), tolerance=1e-12, kwargs...
)::ProjTensorTrain{T} where {T,V}
    return reduce(
        (x, y) -> add(x, y; maxbonddim=maxbonddim, tolerance=tolerance, kwargs...),
        (
            approxtt(x; maxbonddim=maxbonddim, tolerance=tolerance, kwargs...) for
            x in obj.data
        ),
    )
end

function isapproxttavailable(obj::ProjContainer)
    return reduce(&, isapproxttavailable(x) for x in obj.data)
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
    if length(leftmmultiidxset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    result = obj.data[1](leftmmultiidxset, rightmmultiidxset, Val(M))
    for o in obj.data[2:end]
        result .+= o(leftmmultiidxset, rightmmultiidxset, Val(M))
    end

    L = length(obj.sitedims)
    NL = length(leftmmultiidxset[1])
    NR = length(rightmmultiidxset[1])

    @show NL, NR, M
    @show collect(Iterators.flatten(obj.sitedims[(NL + 1):(end - NR)]))
    @show size(result)
    @show length(leftmmultiidxset)
    @show length(rightmmultiidxset)

    results_multii = reshape(
        result,
        size(result)[1],
        Iterators.flatten(obj.sitedims[(NL + 1):(end - NR)])...,
        size(result)[end],
    )

    slice = map(
        x -> x == 0 ? Colon() : 1,
        Iterators.flatten((obj.projector[n] for n in (NL + 1):(L - NR))),
    )

    return_shape = [
        prod(p_ == 0 ? s_ : 1 for (s_, p_) in zip(obj.sitedims[n], obj.projector[n])) for
        n in (NL + 1):(L - NR)
    ]

    results_multii_reduced = results_multii[:, slice..., :]
    @show size(results_multii)
    @show size(results_multii_reduced)
    @show return_shape
    @show obj.projector.data[1:4]
    return reshape(
        results_multii_reduced,
        length(leftmmultiidxset),
        Iterators.flatten(return_shape)...,
        length(rightmmultiidxset),
    )
end

function Base.show(io::IO, obj::ProjContainer{T,V}) where {T,V}
    return print(io, "ProjContainer{$T,$V} with $(length(obj.data)) elements")
end

function Base.show(io::IO, obj::ProjContainer{T,ProjTensorTrain{T}}) where {T}
    return print(io, "ProjTTContainer{$T} with $(length(obj.data)) elements")
end
