using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "Projector" begin
    @testset "constructor" begin
        sitedims = [[3], [3], [3]]
        @test all(Projector([[1], [2], [3]], sitedims).data .== [[1], [2], [3]])
    end

    @testset "copy" begin
        sitedims = [[3], [3], [3]]
        prj = TCIA.Projector([[2],[1],[0]], sitedims)
        @test copy(prj) == prj
    end

    @testset "indexaccess" begin
        sitedims = [[2,2], [2,2], [2,2]]
        prj = TCIA.Projector([[2,1],[1,1],[0,0]], sitedims) 
        @test prj(lastindex(prj), 1) == 0
    end
    @testset "comparison" begin
        sitedims = [[3], [3], [3]]
        @test (
            TCIA.hasoverlap(Projector([[1], [2], [3]], sitedims), Projector([[0], [2], [3]], sitedims))
        ) == true
        @test (
            TCIA.hasoverlap(Projector([[1], [2], [3]], sitedims), Projector([[0], [3], [0]], sitedims))
        ) == false
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
            Projector([[2], [0], [0]], sitedims) >= Projector([[1], [0], [0]], sitedims)
        ) == false
        @test (
            Projector([[1], [2], [3]], sitedims) == Projector([[1], [2], [3]], sitedims)
        ) == true
        @test ([[0], [2], [3]] <= Projector([[1], [2], [3]], sitedims)) == false
    end

    @testset "logical operation" begin
        sitedims = [[3], [3]]
        @test Projector([[1], [2]], sitedims) & Projector([[0], [0]], sitedims) ==
            Projector([[1], [2]], sitedims)

        sitedims = [[3, 3], [3]]
        @test Projector([[1, 0], [2]], sitedims) &
              TCIA.Projector([[0, 3], [2]], sitedims) ==
            TCIA.Projector([[1, 3], [2]], sitedims)

        sitedims = [[3, 3],[3, 3]]
        @test Projector([[1, 0], [2, 1]], sitedims) |
            TCIA.Projector([[1, 3], [2,3]], sitedims) ==
            TCIA.Projector([[1, 0], [2,0]], sitedims)
    end

    @testset "reshape" begin
        sitedims = [[4], [8]]
        sitedimsnew = [[2, 2], [2, 2, 2]]

        @test reshape(Projector([[1], [2]], sitedims), sitedimsnew) ==
            Projector([[1, 1], [2, 1, 1]], sitedimsnew)
        @test reshape(Projector([[0], [2]], sitedims), sitedimsnew) ==
            Projector([[0, 0], [2, 1, 1]], sitedimsnew)
    end

    @testset "projectedshape" begin
        sitedims = [[2, 2], [2, 2], [2, 2]]

        p = TCIA.Projector([[0, 0], [1, 1], [0, 0]], sitedims)
        @test TCIA.projectedshape(p, 1, 3) == [4, 1, 4]

        p = TCIA.Projector([[0, 0], [1, 0], [0, 0]], sitedims)
        @test TCIA.projectedshape(p, 1, 2) == [4, 2]
    end

    @testset "fullindices" begin
        sitedims = [[2, 2], [2, 2], [2, 2]]
        p = TCIA.Projector([[0, 0], [2, 2], [0, 0]], sitedims)
        @test TCIA.fullindices(p, [[1, 1], [1, 1]]) == [[1, 1], [2, 2], [1, 1]]
        @test TCIA.fullindices(p, [1, 1]) == [1, 4, 1]
    end

    @testset "check left and rightmmultiidxset" begin
        N = 3
        sitedims = [[2, 2] for _ in 1:N]

        p = TCIA.Projector([[0, 0], [0, 0], [2, 2]], sitedims)
        @test TCIA.isleftmmultiidx_contained(p, [[1, 1]]) == true
        @test TCIA.isleftmmultiidx_contained(p, [[1, 1], [1, 1], [1, 2]]) == false
        @test TCIA.isrightmmultiidx_contained(p, [[2, 2]]) == true
        @test TCIA.isrightmmultiidx_contained(p, [[1, 2]]) == false
    end
end
