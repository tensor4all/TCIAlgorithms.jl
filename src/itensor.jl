struct ProjMPO
    data::MPO
    sites::Vector{Vector{Index}}
    projector::Projector

    """
    Constructor for ProjMPO.
    The underlying memory layout of the data is permuted to match the order of the site indices.
    The data may be copied.
    """
    function ProjMPO(
        data::MPO, sites::AbstractVector{<:AbstractVector}, projector::Projector
    )
        _check_projector_compatibility(projector, data, sites) || error(
            "Incompatible projector and data. Even small numerical noise can cause this error.",
        )
        return new(permutesiteinds(data, sites), sites, projector)
    end
end

function ProjMPO(Ψ::MPO, sites::AbstractVector{<:AbstractVector})
    sitedims = [collect(dim.(s)) for s in sites]
    globalprojector = Projector([fill(0, length(s)) for s in sitedims], sitedims)
    return ProjMPO(Ψ, sites, globalprojector)
end

function Base.show(io::IO, obj::ProjMPO)
    return print(io, "ProjMPO projected on $(obj.projector.data)")
end

#_permdims(tensor::ITensor, inds...) = ITensor(Array(tensor, inds...), inds...)
#_permdims(tensor::ITensor, inds::AbstractVector) = _permdims(tensor, inds...)

function permutesiteinds(Ψ::MPO, sites::AbstractVector{<:AbstractVector})
    links = linkinds(Ψ)
    tensors = Vector{ITensor}(undef, length(Ψ))
    tensors[1] = permute(Ψ[1], vcat(sites[1], links[1]))
    for n in 2:(length(Ψ) - 1)
        tensors[n] = permute(Ψ[n], vcat(links[n - 1], sites[n], links[n]))
    end
    tensors[end] = permute(Ψ[end], vcat(links[end], sites[end]))
    return MPO(tensors)
end

# Conversion from ProjMPO to MPO
ITensors.MPO(projΨ::ProjMPO) = projΨ.data

# Conversion from ProjMPO to ProjTensorTrain
function ProjTensorTrain{T}(projΨ::ProjMPO) where {T}
    return ProjTensorTrain{T}(asTT3(T, projΨ.data, sites; permdims=false), projΨ.projector)
end

# Conversion from ProjTensorTrain to ProjMPO
function ProjMPO(projtt::ProjTensorTrain{T}) where {T}
    # To be implemented
end

function project(tensor::ITensor, projsiteinds::Dict{K,Int}) where {K}
    slice = Union{Int,Colon}[
        idx ∈ keys(projsiteinds) ? projsiteinds[idx] : Colon() for idx in inds(tensor)
    ]
    data_org = Array(tensor, inds(tensor)...)
    data_trim = zero(data_org)
    data_trim[slice...] .= data_org[slice...]
    return ITensor(data_trim, inds(tensor)...)
end

function find_nested_index(data::Vector{Vector{T}}, target::T) where {T}
    for (i, subvector) in enumerate(data)
        j = findfirst(x -> x == target, subvector)
        if j !== nothing
            return (i, j)
        end
    end
    return nothing  # Not found
end

function project(oldprojector::Projector, sites, projsiteinds::Dict{Index{T},Int}) where {T}
    newprojdata = deepcopy(oldprojector.data)
    for (siteind, projind) in projsiteinds
        pos = find_nested_index(sites, siteind)
        if pos === nothing
            error("Site index not found: $siteind")
        end
        newprojdata[pos[1]][pos[2]] = projind
    end
    return Projector(newprojdata, oldprojector.sitedims)
end

function project(projΨ::ProjMPO, projsiteinds::Dict{Index{T},Int}) where {T}
    return ProjMPO(
        MPO([project(projΨ.data[n], projsiteinds) for n in 1:length(projΨ.data)]),
        projΨ.sites,
        project(projΨ.projector, projΨ.sites, projsiteinds),
    )
end

#==
function asTT3(::Type{T}, Ψ::MPO, sites; permdims=true)::TensorTrain{T,3} where {T}
    Ψ2 = permdims ? _permdims(Ψ, sites) : Ψ
    tensors = Array{T,3}[]
    links = linkinds(Ψ2)
    push!(tensors, reshape(Array(Ψ2[1]), 1, :, links[1]))
    for n in 2:(length(Ψ2) - 1)
        push!(tensors, reshape(Array(Ψ2[n]), links[n - 1], :, links[n]))
    end
    push!(tensors, reshape(Array(Ψ2[end]), links[end - 1], :, 1))
    return TensorTrain{T,3}(tensors)
end
==#

function _check_projector_compatibility(
    projector::Projector, Ψ::MPO, sites::AbstractVector{<:AbstractVector}
)
    links = linkinds(Ψ)
    sitedims = [collect(dim.(s)) for s in sites]

    sitetensors = []
    push!(
        sitetensors,
        reshape(
            Array(Ψ[1], [sites[1]..., links[1]]), vcat(1, sitedims[1], dim(links[1]))...
        ),
    )
    for n in 2:(length(Ψ) - 1)
        push!(sitetensors, Array(Ψ[n], [links[n - 1], sites[n]..., links[n]]))
    end
    push!(
        sitetensors,
        reshape(
            Array(Ψ[end], [links[end], sites[end]...]),
            vcat(dim(links[end]), sitedims[end], 1)...,
        ),
    )

    return reduce(
        &,
        _check_projector_compatibility(projector[n], sitedims[n], sitetensors[n]) for
        n in 1:length(Ψ)
    )
end

struct ProjMPOContainer
    # The projectors can overlap with each other.
    data::Vector{ProjMPO}

    # The site indices of the MPOs in `data`
    # The order of site index vectors in `sites` does not necessarily match the order of the MPOs in `data`.
    sites::Vector{Vector{Index}}

    projector::Projector

    function ProjMPOContainer(data::AbstractVector{ProjMPO})
        for n in 2:length(data)
            data[n].sites == data[1].sites ||
                error("Sitedims mismatch $(data[n].sites) != $(data[1].sites)")
        end
        projector = reduce(|, x.projector for x in data)
        return new(data, data[1].sites, projector)
    end
end

#==
function _random_mpo(
    rng::AbstractRNG, sites::AbstractVector{<:AbstractVector{Index{T}}}; m::Int=1
) where {T}
    sites_ = collect(Iterators.flatten(sites))
    Ψ = random_mps(rng, sites_, m)
    tensors = ITensor[]
    pos = 1
    for i in 1:length(sites)
        push!(tensors, prod(Ψ[pos:(pos + length(sites[i]) - 1)]))
        pos += length(sites[i])
    end
    return MPO(tensors)
end

function _random_mpo(sites::AbstractVector{<:AbstractVector{Index{T}}}; m::Int=1) where {T}
    return _random_mpo(Random.default_rng(), sites; m=m)
end
==#

# Wrappers for
# matmul()
# adaptivematmul()
