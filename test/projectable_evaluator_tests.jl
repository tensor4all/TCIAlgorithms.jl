using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

import QuanticsGrids: DiscretizedGrid, quantics_to_origcoord, origcoord_to_quantics
import QuanticsGrids as QG

@testset "ProjEvaluator" begin
    @testset "makeprojectableMulti" begin
        R = 5
        localdims = fill(4, R)
        sitedims = [[x] for x in localdims]

        qf = TCI.makebatchevaluatable(Float64, x -> sum(x), localdims)
        pqf = TCIA.ProjectableEvaluatorAdapter{Float64}(qf, sitedims)

        leftindexset_ = [[[4], [1], [2]]]
        rightindexset_ = [[[1]]]
        leftindexset = [[4, 1, 2]]
        rightindexset = [[1]]

        @test leftindexset == TCIA._lineari(pqf, leftindexset_, rightindexset_)[1]
        @test rightindexset == TCIA._lineari(pqf, leftindexset_, rightindexset_)[2]

        @test vec(pqf(leftindexset, rightindexset, Val(R - 4))) ≈
            vec([qf([l..., i, r...]) for l in leftindexset, i in 1:4, r in rightindexset])
    end

    @testset "makeprojectableMMulti" begin
        R = 4
        fxy(x, y) = x^2 + y^2
        grid = DiscretizedGrid{2}(R, (-1, -1), (1, 1))
        localdims = fill(4, R)
        qf = x -> fxy(quantics_to_origcoord(grid, x)...)

        pqf = TCIA.makeprojectable(Float64, qf, localdims)

        sitedims = [[x] for x in localdims]

        idx = fill(1, R)
        @test pqf([[i] for i in idx]) ≈ qf(idx)

        leftindexset = [[1]]
        rightmmultiidxset = [[1, 1]]
        leftindexset_ = [[[x] for x in y] for y in leftindexset]
        rightmmultiidxset_ = [[[x] for x in y] for y in rightmmultiidxset]

        @test vec(pqf(leftindexset_, rightmmultiidxset_, Val(R - 3))) ≈ vec([
            qf([l..., i, r...]) for l in leftindexset, i in 1:4, r in rightmmultiidxset
        ])
    end
end
