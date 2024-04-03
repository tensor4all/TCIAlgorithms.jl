using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "indexset" begin
    multii = [[[1, 1]], [[2, 1]]]
    lineari = [[1], [2]]
    sitedims = [[2, 2]]
    for (mi, li) in zip(multii, lineari)
        @test TCIA.lineari(sitedims, mi) == li
        @test TCIA.multii(sitedims, li) == mi
    end
end
