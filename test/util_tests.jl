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

@testset "findinitialpivots" begin
    R = 8
    localdims = fill(2, R)
    f = x -> sum(x)

    pivots = TCIA.findinitialpivots(f, localdims, 10)
    @test length(pivots) == 10
    @test all(f.(pivots) .== sum(localdims))
end

@testset "_contract" begin
    a = rand(2, 3, 4)
    b = rand(2, 5, 4)
    ab = TCIA._contract(a, b, (1, 3), (1, 3))
    @test vec(
        reshape(permutedims(a, (2, 1, 3)), 3, :) * reshape(permutedims(b, (1, 3, 2)), :, 5)
    ) ≈ vec(ab)
end