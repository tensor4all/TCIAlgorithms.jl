using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "ProjContainer" begin
    @testset "ProjTTContainer" begin
        N = 4
        χ = 2
        bonddims = [1, χ, χ, χ, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [2, 2, 2, 2]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
        localdims = collect(prod.(sitedims))

        tt = TCIA.ProjTensorTrain(
            TCI.TensorTrain([
                rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for
                n in 1:N
            ]),
        )

        projs = [
            TCIA.Projector(vcat([[i, j]], [[0, 0] for _ in 1:(N - 1)]), sitedims) for
            i in 1:2, j in 1:2
        ]
        ptts = [TCIA.project(tt, p) for p in vec(projs)]
        pttc = TCIA.ProjTTContainer(ptts)

        for i in 1:2, j in 1:2
            @test pttc(vcat([[i, j]], [[1, 1] for _ in 1:(N - 1)])) ≈ tt(vcat([[i, j]], [[1, 1] for _ in 1:(N - 1)]))
        end
    end
end
