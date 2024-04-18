using Test
using LinearAlgebra
import TensorCrossInterpolation as TCI
import TCIAlgorithms: MatrixProduct
import TCIAlgorithms as TCIA
using TCIITensorConversion

using ITensors

#==
@testset "MatrixProduct (evaluateleft)" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    @assert length(bonddims_a) == N + 1
    @assert length(bonddims_b) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain([
        rand(bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1]) for n = 1:N
    ])
    b = TCI.TensorTrain([
        rand(bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1]) for n = 1:N
    ])

    ab = TCIA.MatrixProduct(a, b)

    for n = 1:N
        for i1 = 1:localdims1[n], i3 = 1:localdims3[n]
            idx_fused = i1 + (i3 - 1) * localdims1[n]
            @test TCIA._unfuse_idx(ab, n, idx_fused) == (i1, i3)
            @test TCIA._fuse_idx(ab, n, (i1, i3)) == idx_fused
        end
    end

    # For testing
    links_a = ab.links_a
    links_b = ab.links_b

    sites1 = ab.sites1
    sites2 = ab.sites2
    sites3 = ab.sites3

    a_MPO = ab.a_MPO
    b_MPO = ab.b_MPO

    for n = 1:N-1
        result = []
        ref = []
        for inds1_ in Iterators.product((1:d for d in localdims1[1:n])...),
            inds3_ in Iterators.product((1:d for d in localdims3[1:n])...)

            inds1 = collect(inds1_)
            inds3 = collect(inds3_)
            inds = collect(zip(inds1, inds3))

            res = ITensor(1)
            res *= onehot(links_a[1] => 1)
            res *= onehot(links_b[1] => 1)
            for i = 1:n
                res *= a_MPO[i] * onehot(sites1[i] => inds1[i])
                res *= b_MPO[i] * onehot(sites3[i] => inds3[i])
            end

            push!(result, TCIA.evaluateleft(ab, inds))
            push!(ref, Array(res, links_a[n+1], links_b[n+1]))
        end
        @test collect(result) ≈ collect(ref)
    end

end

@testset "MatrixProduct (evaluateright)" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    @assert length(bonddims_a) == N + 1
    @assert length(bonddims_b) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain([
        rand(bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1]) for n = 1:N
    ])
    b = TCI.TensorTrain([
        rand(bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1]) for n = 1:N
    ])

    ab = TCIA.MatrixProduct(a, b)

    # For testing
    links_a = ab.links_a
    links_b = ab.links_b

    sites1 = ab.sites1
    sites2 = ab.sites2
    sites3 = ab.sites3

    a_MPO = ab.a_MPO
    b_MPO = ab.b_MPO

    for n = 1:N-1
        result = []
        ref = []
        for inds1_ in Iterators.product((1:d for d in localdims1[N-n+1:N])...),
            inds3_ in Iterators.product((1:d for d in localdims3[N-n+1:N])...)

            inds1 = collect(inds1_)
            inds3 = collect(inds3_)
            inds = collect(zip(inds1, inds3))

            res = ITensor(1)
            res *= onehot(links_a[end] => 1)
            res *= onehot(links_b[end] => 1)
            for i = 1:n
                res *= a_MPO[i+N-n] * onehot(sites1[i+N-n] => inds1[i])
                res *= b_MPO[i+N-n] * onehot(sites3[i+N-n] => inds3[i])
            end

            push!(result, TCIA.evaluateright(ab, inds))
            push!(ref, Array(res, links_a[N-n+1], links_b[N-n+1]))
        end
        @test collect(result) ≈ collect(ref)
    end

end

@testset "MatrixProduct (evaluateright)" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    @assert length(bonddims_a) == N + 1
    @assert length(bonddims_b) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    localdims_fused = localdims1 .* localdims3

    a = TCI.TensorTrain([
        rand(bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1]) for n = 1:N
    ])
    b = TCI.TensorTrain([
        rand(bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1]) for n = 1:N
    ])

    ab = TCIA.MatrixProduct(a, b)

    @test TCI.isbatchevaluable(ab)

    ab_MPO = TCIA._contract(ab)

    ab_ref = Array(reduce(*, ab_MPO), collect(Iterators.flatten(zip(ab.sites1, ab.sites3))))
    ab_ref = reshape(ab_ref, localdims_fused...)

    nrand_inds = 10
    rand_inds = [[rand(1:ld) for ld in localdims_fused] for _ = 1:nrand_inds]

    # Test evaluate
    for inds in rand_inds
        @test TCIA.evaluate(ab, inds) ≈ ab_ref[inds...]
    end

    # Test batchevaluate
    for (nl, nr) in [(0, 2), (2, 0), (1, 1)]
        leftindexset = unique(Vector{Int}[ind[1:nl] for ind in rand_inds])
        rightindexset = unique(Vector{Int}[ind[N-nr+1:end] for ind in rand_inds])

        start_site = nl + 1
        end_site = N - nr

        batcheval_res = TCI.batchevaluate(ab, leftindexset, rightindexset, Val(N - nl - nr))
        for (il, lind) in enumerate(leftindexset)
            for (ir, rind) in enumerate(rightindexset)
                for inds_mid_ in Iterators.product(
                    (1:d for d in localdims_fused[start_site:end_site])...,
                )
                    inds_mid = collect(inds_mid_)
                    inds = vcat(lind, inds_mid, rind)
                    @test batcheval_res[il, inds_mid..., ir] ≈ ab_ref[inds...]
                end
            end
        end
    end
end
==#

function _tomat(tto::TCI.TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(CartesianIndices(Tuple(localdims1)))
        for (j, inds2) in enumerate(CartesianIndices(Tuple(localdims2)))
            mat[i, j] = TCI.evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

@testset "MPO-MPO naive contraction" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1]) for
        n in 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n + 1]) for
        n in 1:N
    ])

    ab = TCIA.naivecontract(a, b)

    sites1 = Index.(localdims1, "1")
    sites2 = Index.(localdims2, "2")
    sites3 = Index.(localdims3, "3")

    #amps = MPO(a, sites = collect(zip(sites1, sites2)))
    #bmps = MPO(b, sites = collect(zip(sites2, sites3)))
    #abmps = amps * bmps

    @test _tomat(ab) ≈ _tomat(a) * _tomat(b)

    #for inds1 in CartesianIndices(Tuple(localdims1))
    #for inds3 in CartesianIndices(Tuple(localdims3))
    #refvalue = evaluate_mps(
    #abmps,
    #collect(zip(sites1, Tuple(inds1))),
    #collect(zip(sites3, Tuple(inds3))),
    #)
    #inds = collect(zip(Tuple(inds1), Tuple(inds3)))
    #@test ab(inds) ≈ refvalue
    #end
    #end
end

@testset "MPO-MPO contraction" for f in [x -> x, x -> 2 * x]
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1]) for
        n in 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n + 1]) for
        n in 1:N
    ])

    ab = TCIA.contract(a, b; f=f)
    @test TCI.sitedims(ab) == [[localdims1[i], localdims3[i]] for i in 1:N]
    @test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
end
