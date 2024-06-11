using ITensors

function _test_projection(obj::TCIA.ProjectableEvaluator, prj)
    pobj = TCIA.project(obj, prj)

    # Within the projection
    pobj_full = TCIA.fulltensor(pobj; fused=false)
    obj_full = TCIA.fulltensor(obj; fused=false)

    # Projection
    mask = [x == 0 ? Colon() : x for x in Iterators.flatten(prj.data)]
    @test pobj_full[mask...] ≈ obj_full[mask...]

    let
        tmp = deepcopy(pobj_full)
        tmp[mask...] .= 0.0
        @test all(tmp .== 0.0)
    end

    #indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
    #@test indexset1 <= pobj.projector
    #@test obj(indexset1) == pobj(indexset1) # exact equality

    # Outside the partition
    #indexset2 = [[2, 1], [1, 1], [1, 1], [1, 1]]
    #@test pobj(indexset2) == 0.0

    ## Evaluation at a single linear indexset
    #indexset3 = [[1, 1], [1, 1], [1, 1], [2, 1]]
    #indexset3_li = [1, 1, 1, 2]
    #@test pobj(indexset3) == pobj(indexset3_li)
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
