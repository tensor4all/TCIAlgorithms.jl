using Test
import TensorCrossInterpolation as TCI
using TCIAlgorithms
import TCIAlgorithms as TCIA

@testset "DTensorTrain" begin
    @testset "init" begin
        tt = TCIA.DTensorTrain([rand(2, 3, 5, 4), rand(4, 3, 2, 4), rand(4, 2, 5, 4)])
        @test tt.sitedims == [[3, 5], [3, 2], [2, 5]]

        tt_ref = TCI.TensorTrain([
            reshape(x, size(x, 1), :, size(x)[end]) for x in tt.sitetensors
        ])

        leftindexset = [[[1, 1]], [[1, 2]]]
        rightindexset = [[[1, 1]], [[1, 2]]]

        NL = 1
        NR = 1
        leftindexset_ = [TCIA.lineari(tt.sitedims[1:NL], x) for x in leftindexset]
        rightindexset_ = [
            TCIA.lineari(tt.sitedims[(end - NR + 1):end], x) for x in rightindexset
        ]

        @test tt(leftindexset, rightindexset, Val(1)) ≈
            tt_ref(leftindexset_, rightindexset_, Val(1))
    end
end
