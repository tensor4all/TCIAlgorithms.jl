struct ProjMPS
    data::MPS
    sites::Vector{Vector{Index}}
    projector::Projector

    """
    Constructor for ProjMPS.
    The underlying memory layout of the data is permuted to match the order of the site indices.
    The data may be copied.
    """
    function ProjMPS(
        data::MPS, sites::AbstractVector{<:AbstractVector}, projector::Projector
    )
        _check_projector_compatibility(projector, data, sites) || error(
            "Incompatible projector and data. Even small numerical noise can cause this error.",
        )
        return new(permutesiteinds(data, sites), sites, projector)
    end
end

function ProjMPS(Ψ::MPS, sites::AbstractVector{<:AbstractVector})
    sitedims = [collect(dim.(s)) for s in sites]
    globalprojector = Projector([fill(0, length(s)) for s in sitedims], sitedims)
    return ProjMPS(Ψ, sites, globalprojector)
end

function Base.show(io::IO, obj::ProjMPS)
    return print(io, "ProjMPS projected on $(obj.projector.data)")
end

#_permdims(tensor::ITensor, inds...) = ITensor(Array(tensor, inds...), inds...)
#_permdims(tensor::ITensor, inds::AbstractVector) = _permdims(tensor, inds...)

function permutesiteinds(Ψ::MPS, sites::AbstractVector{<:AbstractVector})
    links = linkinds(Ψ)
    tensors = Vector{ITensor}(undef, length(Ψ))
    tensors[1] = permute(Ψ[1], vcat(sites[1], links[1]))
    for n in 2:(length(Ψ) - 1)
        tensors[n] = permute(Ψ[n], vcat(links[n - 1], sites[n], links[n]))
    end
    tensors[end] = permute(Ψ[end], vcat(links[end], sites[end]))
    return MPS(tensors)
end

# Conversion from ProjMPS to MPS
ITensors.MPS(projΨ::ProjMPS) = projΨ.data

# Conversion from ProjMPS to ProjTensorTrain
function ProjTensorTrain{T}(projΨ::ProjMPS) where {T}
    return ProjTensorTrain{T}(
        asTT3(T, projΨ.data, projΨ.sites; permdims=false), projΨ.projector
    )
end

# Conversion from ProjTensorTrain to ProjMPS
function ProjMPS(::Type{T}, projtt::ProjTensorTrain{T}, sites) where {T}
    # To be implemented
    links = [Index(ld, "Link,l=$l") for (l, ld) in enumerate(TCI.linkdims(projtt.data))]

    tensors = ITensor[]
    sitedims = [collect(dim.(s)) for s in sites]
    linkdims = dim.(links)

    push!(
        tensors,
        ITensor(
            reshape(projtt.data[1], 1, prod(sitedims[1]), linkdims[1]),
            sites[1]...,
            links[1],
        ),
    )

    for n in 2:(length(projtt.data) - 1)
        push!(
            tensors,
            ITensor(
                reshape(projtt.data[n], linkdims[n - 1], prod(sitedims[n]), linkdims[n]),
                links[n - 1],
                sites[n]...,
                links[n],
            ),
        )
    end

    push!(
        tensors,
        ITensor(
            reshape(projtt.data[end], linkdims[end], prod(sitedims[end])),
            links[end],
            sites[end]...,
        ),
    )

    return ProjMPS(MPS(tensors), sites, projtt.projector)
end

Base.isapprox(x::ProjMPS, y::ProjMPS; kwargs...) = Base.isapprox(x.data, y.data, kwargs...)

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

function project(projΨ::ProjMPS, projsiteinds::Dict{Index{T},Int}) where {T}
    return ProjMPS(
        MPS([project(projΨ.data[n], projsiteinds) for n in 1:length(projΨ.data)]),
        projΨ.sites,
        project(projΨ.projector, projΨ.sites, projsiteinds),
    )
end

function asTT3(::Type{T}, Ψ::MPS, sites; permdims=true)::TensorTrain{T,3} where {T}
    Ψ2 = permdims ? _permdims(Ψ, sites) : Ψ
    tensors = Array{T,3}[]
    links = linkinds(Ψ2)
    push!(tensors, reshape(Array(Ψ2[1], sites[1]..., links[1]), 1, :, dim(links[1])))
    for n in 2:(length(Ψ2) - 1)
        push!(
            tensors,
            reshape(
                Array(Ψ2[n], links[n - 1], sites[n]..., llinks[n]),
                dim(links[n - 1]),
                :,
                dim(links[n]),
            ),
        )
    end
    push!(
        tensors, reshape(Array(Ψ2[end], links[end], sites[end]...), dim(links[end]), :, 1)
    )
    return TensorTrain{T,3}(tensors)
end

function _check_projector_compatibility(
    projector::Projector, Ψ::MPS, sites::AbstractVector{<:AbstractVector}
)
    links = linkinds(Ψ)
    sitedims = [collect(dim.(s)) for s in sites]

    sitetensors = []
    push!(
        sitetensors,
        reshape(
            Array(Ψ[1], [sites[1]..., links[1]]), [1, prod(sitedims[1]), dim(links[1])]...
        ),
    )
    for n in 2:(length(Ψ) - 1)
        push!(
            sitetensors,
            reshape(
                Array(Ψ[n], [links[n - 1], sites[n]..., links[n]]),
                dim(links[n - 1]),
                prod(sitedims[n]),
                dim(links[n]),
            ),
        )
    end
    push!(
        sitetensors,
        reshape(
            Array(Ψ[end], [links[end], sites[end]...]),
            dim(links[end]),
            prod(sitedims[end]),
            1,
        ),
    )

    return reduce(
        &,
        _check_projector_compatibility(projector[n], sitedims[n], sitetensors[n]) for
        n in 1:length(Ψ)
    )
end

struct ProjMPSContainer
    # The projectors can overlap with each other.
    data::Vector{ProjMPS}

    # The site indices of the MPSs in `data`
    # The order of site index vectors in `sites` does not necessarily match the order of the MPSs in `data`.
    sites::Vector{Vector{Index}}

    projector::Projector

    function ProjMPSContainer(data::AbstractVector{ProjMPS})
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
    return MPS(tensors)
end

function _random_mpo(sites::AbstractVector{<:AbstractVector{Index{T}}}; m::Int=1) where {T}
    return _random_mpo(Random.default_rng(), sites; m=m)
end
==#

# Wrappers for
# matmul()
# adaptivematmul()
#
#function combinesites(M::MPO, site1::Index, site2::Index)
#end
#function preprocess(mul::Quantics.MatrixMultiplier{T}, M1::ProjMPSContainer, M2::ProjMPSContainer) where {T}
    #for (site_row, site_shared, site_col) in zip(mul.sites_row, mul.sites_shared,
        #mul.sites_col)
        #M1, M2 = combinesites(M1, site_row, site_shared), combinesites(M2, site_col, site_shared)
    #end
    #return M1, M2
#end

function automul_preprocess(sites_rows, sites_shared, sites_cols, M1::ProjMPSContainer, M2::ProjMPSContainer)
    #for (site_row, site_shared, site_col) in zip(sites_rows, sites_shared, sites_cols)
        #M1, M2 = combinesites(M1, site_row, site_shared), combinesites(M2, site_col, site_shared)
    #end
    #return M1, M2
end

"""
Equivalent to `Quantics.automul`
"""
function automul(M1::ProjMPSContainer, M2::ProjMPSContainer; tag_row::String="", tag_shared::String="",
    tag_col::String="", alg="naive", kwargs...)
    if in(:maxbonddim, keys(kwargs))
        error("Illegal keyward parameter: maxbonddim. Use maxdim instead!")
    end
    length.(M1.sites) .== 1 || error("automul only supports MPSs with one site index per site")
    length.(M2.sites) .== 1 || error("automul only supports MPSs with one site index per site")

    sites_row = Quantics.findallsiteinds_by_tag(siteinds(M1[1]); tag=tag_row)
    sites_shared = Quantics.findallsiteinds_by_tag(siteinds(M1[1]); tag=tag_shared)
    sites_col = Quantics.findallsiteinds_by_tag(siteinds(M2[1]); tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    if sites_shared != Quantics.findallsiteinds_by_tag(siteinds(M2[1]); tag=tag_shared)
        error("Invalid shared sites for MatrixMultiplier")
    end

    #matmul = Quantics.MatrixMultiplier(sites_row, sites_shared, sites_col)
    #ewmul = Quantics.ElementwiseMultiplier([s for s in siteinds(M1) if s ∉ sites_matmul])

    #M1_tensors = [Quantics.preprocess(matmul, Quantics.asMPO(x)) for x in M1]
    #M1_ = Quantics.asMPO(M1)
    #M2_ = Quantics.asMPO(M2)
    #M1_, M2_ = preprocess(matmul, M1_, M2_)
    #M1_, M2_ = preprocess(ewmul, M1_, M2_)


    sites1_matmul = Index{Int}[]
    sites1_org = siteinds(M1[1])
    while !isempty(sites1_org)
        s = sites1_org[1]
        if s ∈ sites_matmul
            if s ∈ sites_shared
                push!(sites1_matmul, reverse(sites1_org[1:2]))
                popfirst!(sites1_org)
                popfirst!(sites1_org)
            else
                push!(sites1_matmul, sites1_org[1:2])
                popfirst!(sites1_org)
                popfirst!(sites1_org)
            end
        else
            Quantics.makediagram(s)
        end
    end

    matmul = MatrixMultiplier(sites_row, sites_shared, sites_col)
    ewmul = ElementwiseMultiplier([s for s in siteinds(M1) if s ∉ sites_matmul])

    M1_ = Quantics.asMPO(M1)
    M2_ = Quantics.asMPO(M2)
    M1_, M2_ = preprocess(matmul, M1_, M2_)
    M1_, M2_ = preprocess(ewmul, M1_, M2_)

    M = FastMPOContractions.contract_mpo_mpo(M1_, M2_; alg=alg, kwargs...)

    M = Quantics.postprocess(matmul, M)
    M = Quantics.postprocess(ewmul, M)

    if in(:maxdim, keys(kwargs))
        truncate!(M; maxdim=kwargs[:maxdim])
    end

    return asMPS(M)
end