using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "BlockStructure" begin
    @testset "constructor" begin
        sitedims = [[2], [2], [2]]

        let
            p1 = TCIA.Projector([[0], [0], [0]], sitedims)
            @test_throws ErrorException TCIA.BlockStructure([p1, p1])
        end

        let
            p1 = TCIA.Projector([[1], [0], [0]], sitedims)
            p2 = TCIA.Projector([[2], [0], [0]], sitedims)
            bs = TCIA.BlockStructure([p1, p2])
            @test length(bs) == 2
        end
    end

    @testset "iterator" begin
        sitedims = [[2], [2], [2]]

        p1 = TCIA.Projector([[1], [0], [0]], sitedims)
        p2 = TCIA.Projector([[2], [0], [0]], sitedims)
        bs = TCIA.BlockStructure([p1, p2])
        @test all(collect(bs) .== [p1, p2])
    end
end
