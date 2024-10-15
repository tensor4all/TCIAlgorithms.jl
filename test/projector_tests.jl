using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "Projector" begin
    @testset "constructor" begin
        inds = [Index(2, "n=$n") for n in 1:10]

        @test_throws ErrorException Projector(Dict(inds[1] => 0))
        @test_throws ErrorException Projector(Dict(inds[1] => -1))
    end

    @testset "comparison" begin
        inds = [Index(2, "n=$n") for n in 1:10]

        p1 = Projector(Dict(inds[1] => 1))
        p2 = Projector(Dict(inds[1] => 2))
        p3 = Projector(Dict(inds[1] => 1, inds[2] => 1))

        @test (p1 == p1) == true
        @test (p2 == p2) == true
        @test (p3 == p3) == true
        @test (p1 < p2) == false
        @test (p1 > p2) == false
        @test (p3 < p1) == true
    end

    @testset "intersection" begin
        inds = [Index(2, "n=$n") for n in 1:10]

        let
            p1 = Projector(Dict(inds[1] => 1, inds[2] => 1))
            p2 = Projector(Dict(inds[1] => 2))
            @test_throws ErrorException p1 & p2
        end

        let
            p1 = Projector(Dict(inds[2] => 1))
            p2 = Projector(Dict(inds[1] => 2))
            @test p1 & p2 == Projector(Dict(inds[1] => 2, inds[2] => 1))
        end

        let
            p1 = Projector(Dict(inds[2] => 1))
            p2 = Projector(Dict(inds[1] => 2))
            @test p1 & p2 == Projector(Dict(inds[1] => 2, inds[2] => 1))
        end

        let
            p1 = Projector(Dict(inds[2] => 1, inds[3] => 1))
            p2 = Projector(Dict(inds[1] => 2, inds[3] => 1))
            @test p1 & p2 == Projector(Dict(inds[1] => 2, inds[2] => 1, inds[3] => 1))
        end
    end

    @testset "union" begin
        inds = [Index(2, "n=$n") for n in 1:10]

        p1 = Projector(Dict(inds[1] => 1, inds[3] => 1))
        p2 = Projector(Dict(inds[1] => 2, inds[3] => 1))
        @test p1 | p2 == Projector(Dict(inds[3] => 1))

        p1 = Projector()
        p2 = Projector()
        @test p1 | p2 == Projector()
    end
end
