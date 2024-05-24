using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "Projector" begin
    @testset "constructor" begin
        sitedims = [[3], [3], [3]]
        @test all(Projector([[1], [2], [3]], sitedims).data .== [[1], [2], [3]])
    end

    @testset "comparison" begin
        sitedims = [[3], [3], [3]]
        @test (
            Projector([[1], [2], [3]], sitedims) <= Projector([[0], [2], [3]], sitedims)
        ) == true
        @test (
            Projector([[1], [2], [3]], sitedims) < Projector([[0], [2], [3]], sitedims)
        ) == true
        @test (
            Projector([[1], [2], [3]], sitedims) <= Projector([[1], [2], [3]], sitedims)
        ) == true
        @test (
            Projector([[1], [0], [0]], sitedims) <= Projector([[2], [0], [0]], sitedims)
        ) == false
        @test (
            Projector([[1], [2], [3]], sitedims) == Projector([[1], [2], [3]], sitedims)
        ) == true
        @test ([[1], [2], [3]] <= Projector([[0], [2], [3]], sitedims)) == true
    end

    @testset "logical operation" begin
        sitedims = [[3], [3]]
        @test Projector([[1], [2]], sitedims) & Projector([[0], [0]], sitedims) ==
            Projector([[1], [2]], sitedims)

        sitedims = [[3, 3], [3]]
        @test Projector([[1, 0], [2]], sitedims) &
              TCIA.Projector([[0, 3], [0]], sitedims) ==
            TCIA.Projector([[1, 3], [2]], sitedims)
    end

    @testset "reshape" begin
        sitedims = [[4], [8]]
        sitedimsnew = [[2, 2], [2, 2, 2]]

        @test reshape(Projector([[1], [2]], sitedims), sitedimsnew) ==
            Projector([[1, 1], [2, 1, 1]], sitedimsnew)
        @test reshape(Projector([[0], [2]], sitedims), sitedimsnew) ==
            Projector([[0, 0], [2, 1, 1]], sitedimsnew)
    end

    @testset "isprojectedat" begin
        sitedims = [[2, 2], [2, 2], [2, 2]]
        p = TCIA.Projector([[0, 0], [1, 1], [0, 0]], sitedims)
        @test [TCIA.isprojectedat(p, n) for n in 1:length(p)] == [false, true, false]

        p = TCIA.Projector([[0, 1]], sitedims)
        @test_throws ErrorException TCIA.isprojectedat(p, 1)
    end

    @testset "fullindices" begin
        sitedims = [[2, 2], [2, 2], [2, 2]]
        p = TCIA.Projector([[0, 0], [2, 2], [0, 0]], sitedims)
        @test [TCIA.isprojectedat(p, n) for n in 1:length(p)] == [false, true, false]
        @test TCIA.fullindices(p, [[1, 1], [1, 1]]) == [[1, 1], [2, 2], [1, 1]]
        @test TCIA.fullindices(p, [1, 1]) == [1, 4, 1]
    end
end
