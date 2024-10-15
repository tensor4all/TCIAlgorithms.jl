"""
Equivalent to Quantics.automul
"""

#==
function automul(M1::ProjMPS, M2::ProjMPS, tag_row::String="", tag_shared::String="",
    tag_col::String="", alg="naive", kwargs...)

    sites_row = findallsiteinds_by_tag(siteinds(M1); tag=tag_row)
    sites_shared = findallsiteinds_by_tag(siteinds(M1); tag=tag_shared)
    sites_col = findallsiteinds_by_tag(siteinds(M2); tag=tag_col)
    sites_matmul = Set(Iterators.flatten([sites_row, sites_shared, sites_col]))

    if sites_shared != findallsiteinds_by_tag(siteinds(M2); tag=tag_shared)
        error("Invalid shared sites for MatrixMultiplier")
    end

    matmul = MatrixMultiplier(sites_row, sites_shared, sites_col)
    ewmul = ElementwiseMultiplier([s for s in siteinds(M1) if s ∉ sites_matmul])

    M1, M2 = preprocess(matmul, M1, M2)
    M1, M2 = preprocess(ewmul, M1, M2)

    M = FastMPOContractions.contract_mpo_mpo(M1, M2; alg=alg, kwargs...)

end
==#

function ITensors.contract(
    M1::ProjMPS, M2::ProjMPS; cutoff=0.0, maxdim=typemax(Int), kwargs...
)
    sites1 = siteinds(M1)
    sites2 = siteinds(M2)

    sites_shared = [intersect(x, y) for (x, y) in zip(sites1, sites2)]
    sites1_external = [setdiff(x, y) for (x, y) in zip(sites1, sites_shared)]
    sites2_external = [setdiff(x, y) for (x, y) in zip(sites2, sites_shared)]

    sites1_perm = [vcat(x, y) for (x, y) in zip(sites1_external, sites_shared)]
    sites2_perm = [vcat(x, y) for (x, y) in zip(sites_shared, sites2_external)]

    T1 = reduce(promote_type, eltype.(M1.data))
    T2 = reduce(promote_type, eltype.(M2.data))
    T = promote_type(T1, T2)

    ptt1 = ProjTensorTrain{T}(permutesiteinds(M1, sites1_perm))
    ptt2 = ProjTensorTrain{T}(permutesiteinds(M2, sites2_perm))

    newproj_data = [
        vcat(p1[1:length(s1)], p2[1:length(s2)]) for (s1, s2, p1, p2) in
        zip(sites1_external, sites2_external, ptt1.projector, ptt2.projector)
    ]

    ptt1 = reshape(
        ptt1,
        [[prod(dim.(x)), prod(dim.(y))] for (x, y) in zip(sites1_external, sites_shared)],
    )
    ptt2 = reshape(
        ptt2,
        [[prod(dim.(x)), prod(dim.(y))] for (x, y) in zip(sites_shared, sites2_external)],
    )

    @show ptt1.projector
    @show ptt2.projector

    ptt12 = approxtt(
        lazymatmul(ptt1, ptt2); maxbonddim=maxdim, tolerance=sqrt(cutoff), kwargs...
    )

    sites12 = [vcat(x, y) for (x, y) in zip(sites1_external, sites2_external)]

    @show newproj_data

    new_sitedims = [dim.(s) for s in sites12]
    new_projector = Projector(newproj_data, new_sitedims)

    ptt12 = reshape(ptt12, new_sitedims)
    @show new_projector

    return ProjMPS(project(ptt12, new_projector), sites12)
end
