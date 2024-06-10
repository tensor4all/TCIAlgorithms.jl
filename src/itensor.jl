struct ProjMPOContainer
    data::Vector{MPO} # Projectors of `data` can overlap with each other
    # The site indices of the MPOs in `data`
    # The order of site index vectors in `sites` does not necessarily match the order of the MPOs in `data`.
    sites::Vector{Vector{Index}}
    projectors::Vector{Projector}

    function ProjMPOContainer(
        data::AbstractVector{<:AbstractMPS},
        sites::AbstractVector{<:AbstractVector},
        projectors::Projector,
    )
        #sitedims = [collect(dim.(x)) for x in sites]
        #_to_MPO(x) = MPO([x _ for x_ in x])
        mpos = MPO[]
        for Ψ in data
            #for (n, sites_n) in enumerate(ITensors.siteinds(Ψ))
            #Set(sites_n) == Set(sites[n]) || error("sites mismatch: $(sites_n) != $(sites[n])")
            #end
            push!(mpos, _to_MPO(Ψ, sites))
        end

        # Check consistency between data and projectors

        #projector = reduce(|, x.projector for x in data)
        #return new{T,V}(data, sitedims, projector)

        return new(data, sites, projectors)
    end
end

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

# Wrappers for
# matmul()
# adaptivematmul()
