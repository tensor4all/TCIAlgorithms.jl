using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA


@testset "allequal" begin
    @test TCIA.allequal([[2,2],[2,2]]) == true
    @test TCIA.allequal([]) == true
end

@testset "indexset" begin
    multii = [[[1, 1]], [[2, 1]], [[0,0]]]
    lineari = [[1], [2], [0]]
    sitedims = [[2, 2]] 
    for (mi, li) in zip(multii, lineari)
        @test TCIA.lineari(sitedims, mi) == li
        @test TCIA.multii(sitedims, li) == mi
    end
end

@testset "Not" begin
    A = [1, 2, 3]
    @test A[collect(TCIA.Not(1, 3))] == A[2:3]
    @test A[collect(TCIA.Not(2, 3))] == [A[1], A[3]]
end

@testset "Iterator" begin
    sitedims = [2, 2]
    A = Array{Tuple{Int, Int}, 2}(undef, Tuple(sitedims))
    for i in CartesianIndices(Tuple(sitedims))
        A[i] = Tuple(i)
    end
    @test A == collect(TCIA.typesafe_iterators_product(Val(2), sitedims)) 
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

@testset "shallowcopy" begin
    prj = TCIA.Projector([[1],[2],[0],[0]],[[2],[2],[2],[2]])
    @test TCIA.shallowcopy(prj).data == prj.data
end