using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "ProjectableEvaluator" begin
    @testset "Wrapper" begin
        localdims1 = [2, 2, 2]
        localdims2 = [2, 2, 2]
        sitedims = [[2, 2], [2, 2], [2, 2]]

        N = length(sitedims)
        bonddims = [1, 4, 4, 1]
        @assert length(bonddims) == N + 1

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])

        p = TCIA.Projector([[0, 0], [2, 2], [0, 0]], sitedims)

        ptt = TCIA.ProjectedTensorTrain(tt, p)

        ptt_wrapper = TCIA._FuncAdapterTCI2Subset(ptt)
        @test ptt_wrapper([[1, 1], [1, 1]]) ≈ ptt([[1, 1], [2, 2], [1, 1]])
        @test ptt_wrapper([1, 1]) ≈ ptt([1, 4, 1])
        @test ptt_wrapper.localdims == [4, 4]
    end
end