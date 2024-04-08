using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "indexset" begin
    multii = [
        [[1, 1]],
        [[2, 1]]
    ]
    lineari = [
        [1],
        [2]
    ]
    sitedims = [[2, 2]]
    for (mi, li) in zip(multii, lineari)
        @test TCIA.lineari(sitedims, mi) == li
        @test TCIA.multii(sitedims, li) == mi
    end
end

@testset "findinitialpivots" begin
    R = 8
    localdims = fill(2, R)
    f = x -> sum(x)

    pivots = TCIA.findinitialpivots(f, localdims, 10)
    @test length(pivots) == 10
    @test all(f.(pivots) .== sum(localdims))
end