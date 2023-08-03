using Test
using LinearAlgebra
import TensorCrossInterpolation as TCI
import TCIAlgorithms: MatrixProduct
import TCIAlgorithms as TCIA

using ITensors

@testset "MPO-MPO contraction" begin
    N = 4
    bonddims_a = [1, 2, 3, 2, 1]
    bonddims_b = [1, 2, 3, 2, 1]
    @assert length(bonddims_a) == N + 1
    @assert length(bonddims_b) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]
    localdims3 = [2, 2, 2, 2]

    a = TCI.TensorTrain([rand(bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1]) for n = 1:N])
    b = TCI.TensorTrain([rand(bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1]) for n = 1:N])

    ab = TCIA.MatrixProduct(a, b)

    for n in 1:N
        for i1 in 1:localdims1[n], i3 in 1:localdims3[n]
            idx_fused = i1 + (i3-1) * localdims1[n] 
            @test TCIA._unfuse_idx(ab, n, idx_fused) == (i1, i3)
            @test TCIA._fuse_idx(ab, n, (i1, i3)) == idx_fused
        end
    end

    for i1 in 1:localdims1[1], i3 in 1:localdims3[1]
        idx_fused = TCIA._fuse_idx(ab, 1, (i1, i3))
        @test TCIA.evaluateleft(ab, [(i1,i3)]) ≈ transpose(a.T[1][1, i1, :, :]) * b.T[1][1, :, i3, :]
    end

    # For testing
    links_a = [Index(bonddims_a[n], "Link=$n") for n in 1:N+1]
    links_b = [Index(bonddims_b[n], "Link=$n") for n in 1:N+1]

    sites1 = [Index(localdims1[n], "Site=$n") for n in 1:N]
    sites2 = [Index(localdims2[n], "Site=$n") for n in 1:N]
    sites3 = [Index(localdims3[n], "Site=$n") for n in 1:N]

    a_MPO = MPO([ITensor(a.T[n], links_a[n], sites1[n], sites2[n], links_a[n+1]) for n in 1:N])
    b_MPO = MPO([ITensor(b.T[n], links_b[n], sites2[n], sites3[n], links_b[n+1]) for n in 1:N])

    for n in 1:N-1
        result = []
        ref = []
        for inds1_ in Iterators.product((1:d for d in localdims1[1:n])...), inds3_ in Iterators.product((1:d for d in localdims3[1:n])...)
            inds1 = collect(inds1_)
            inds3 = collect(inds3_)
            inds = collect(zip(inds1, inds3))

            res = ITensor(1)
            res *= onehot(links_a[1]=>1)
            res *= onehot(links_b[1]=>1)
            for i in 1:n
                res *= a_MPO[i] * onehot(sites1[i]=>inds1[i])
                res *= b_MPO[i] * onehot(sites3[i]=>inds3[i])
            end
    
            push!(result, TCIA.evaluateleft(ab, inds))
            push!(ref, Array(res, links_a[n+1], links_b[n+1]))
        end
        @test collect(result) ≈ collect(ref)
    end

end