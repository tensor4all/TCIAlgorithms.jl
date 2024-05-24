using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "projectat!" begin
    A_org = ones(2, 2, 2)

    let
        A = copy(A_org)
        TCIA.projectat!(A, 1, 1)
        @test all(A[1, :, :] .== A_org)
        @test all(A[2, :, :] .== 0.0)
    end

    let
        A = copy(A_org)
        TCIA.projectat!(A, 1, 2)
        @test all(A[1, :, :] .== 0.0)
        @test all(A[2, :, :] .== A_org)
    end

    let
        A = copy(A_org)
        TCIA.projectat!(A, 2, 1)
        @test all(A[:, 1, :] .== A_org)
        @test all(A[:, 2, :] .== 0.0)
    end
end

@testset "truncate" begin
    using Random
    Random.seed!(100)
    N = 4
    bonddims = [1, 30, 30, 30, 1]
    @assert length(bonddims) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [2, 2, 2, 2]

    tt = TCI.TensorTrain([
        rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
    ])

    tt2 = TCIA.truncate(tt; cutoff=1e-30)

    indices = [[[rand(1:localdims1[n]), rand(1:localdims2[n])] for n in 1:N] for _ in 1:100]
    before = [tt(idx) for idx in indices]
    after = [tt2(idx) for idx in indices]
    @test before ≈ after
end

@testset "ProjectedTensorTrain" begin
    N = 4
    bonddims = [1, 10, 10, 10, 1]
    @assert length(bonddims) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [3, 3, 3, 3]

    tt = TCI.TensorTrain([
        rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
    ])

    # Projection 
    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
    globalprj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
    prj = TCIA.Projector([[1, 0], [0, 0], [0, 0], [0, 0]], sitedims)
    prj_data = deepcopy(TCI.sitetensors(tt))
    prj_data[1][:, 2, :, :] .= 0.0
    ptt = TCIA.ProjectedTensorTrain{Float64,4}(tt, prj)

    # Within the partition
    indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
    @test indexset1 <= ptt.projector
    @test tt(indexset1) == ptt(indexset1) # exact equality

    # Outside the partition
    indexset2 = [[2, 1], [1, 1], [1, 1], [1, 1]]
    @test ptt(indexset2) == 0.0

    # Evaluation at a single linear indexset
    indexset3 = [[1, 1], [1, 1], [1, 1], [2, 1]]
    indexset3_li = [1, 1, 1, 2]
    @test ptt(indexset3) == ptt(indexset3_li)

    # Projection with truncation
    ptt_truncated = TCIA.ProjectedTensorTrain{Float64,4}(tt, globalprj)
    ptt_truncated = TCIA.project(ptt_truncated, prj; compression=true)
    indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
    @test tt(indexset1) ≈ ptt_truncated(indexset1) # exact equality
end

@testset "ProjectedTensorTrain <= TensorTrain" begin
    N = 4
    bonddims = [1, 10, 10, 10, 1]
    @assert length(bonddims) == N + 1

    localdims1 = [1, 1, 1, 1]
    localdims2 = [1, 1, 1, 1]

    tt = TCI.TensorTrain([
        rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
    ])

    ptt = TCIA.ProjectedTensorTrain(tt)

    @test tt([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈ ptt([[1, 1], [1, 1], [1, 1], [1, 1]])
end

@testset "batchevaluateprj" begin
    sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]

    N = length(sitedims)
    bonddims = [1, 4, 4, 4, 1]
    @assert length(bonddims) == N + 1

    tt = TCI.TensorTrain([rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N])

    p = TCIA.Projector([[0, 0], [2, 2], [0, 0], [0, 0]], sitedims)

    ptt = TCIA.ProjectedTensorTrain(tt, p; compression=true)

    leftindexset = [[[1, 1]]]
    rightindexset = [[[1, 1]]]
    batchprj = TCIA.batchevaluateprj(ptt, leftindexset, rightindexset, Val(2))

    @assert size(batchprj) == (1, 1, 4, 1)
    @test batchprj[1, 1, 1, 1] ≈ ptt([1, 4, 1, 1])
end