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

@testset "PartitionedTensorTrain" for compression in [false, true]
    N = 4
    bonddims = [1, 10, 10, 10, 1]
    @assert length(bonddims) == N + 1

    localdims1 = [2, 2, 2, 2]
    localdims2 = [2, 2, 2, 2]

    tt = TCI.TensorTrain([
        rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
    ])

    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
    outer_prj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)

    ptt = TCIA.PartitionedTensorTrain(TCIA.ProjectedTensorTrain(tt, outer_prj))

    ptt2 = TCIA.partitionat(ptt, 1; compression=compression)
    ref = [[1, 1], [2, 1], [1, 2], [2, 2]]
    for (i, obj) in enumerate(ptt2.tensortrains)
        @test obj.projector[1] == ref[i]
    end

    ptt3 = TCIA.partitionat(ptt2, N; compression=compression)
    idx = 1
    for i in ref, j in ref
        @test ptt3.tensortrains[idx].projector[1] == i
        @test ptt3.tensortrains[idx].projector[N] == j
        idx += 1
    end
end

@testset "ProjectedTensorTrainProduct" begin
    N = 4
    bonddims = [1, 10, 10, 10, 1]

    localdims1 = [2, 2, 2, 2]
    localdims2 = [2, 2, 2, 2]
    sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]

    ptt1 = TCIA.ProjectedTensorTrain(
        TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ]),
        TCIA.Projector([[1, 0], [0, 1], [0, 0], [0, 0]], sitedims),
    )
    ptt2 = TCIA.ProjectedTensorTrain(
        TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ]),
        TCIA.Projector([[0, 1], [0, 0], [0, 0], [0, 0]], sitedims),
    )

    pprod = TCIA.create_projected_tensortrain_product((ptt1, ptt2))
    @test pprod !== nothing

    @test pprod.projector == TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)

    ref = TCIA.MatrixProduct(ptt1.data, ptt2.data)

    @test ref([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈ pprod([[1, 1], [1, 1], [1, 1], [1, 1]])
    @test ptt2([[1, 2], [1, 1], [1, 1], [1, 1]]) == 0.0

    @test ref([[1, 2], [1, 1], [1, 1], [1, 1]]) == 0.0
    @test pprod([[1, 2], [1, 1], [1, 1], [1, 1]]) == 0.0

    leftindexset = [[1]]
    rightindexset = [[1]]
    @test pprod(leftindexset, rightindexset, Val(2)) ≈
        ref(leftindexset, rightindexset, Val(2))

    multiplier = TCIA.create_multiplier([ptt1], [ptt2], pprod.projector)
    @test multiplier([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈
        ref([[1, 1], [1, 1], [1, 1], [1, 1]])

    let
        leftindexsets = [[1], [2]]
        rightindexsets = [[1], [2]]
        @test multiplier(leftindexsets, rightindexsets, Val(2)) ≈
            ref(leftindexsets, rightindexsets, Val(2))
    end

    let
        leftindexsets = [[1, 1], [2, 1], [1, 2], [2, 2]]
        rightindexsets = [[1], [2]]
        @test multiplier(leftindexsets, rightindexsets, Val(1)) ≈
            ref(leftindexsets, rightindexsets, Val(1))
    end
end
