using Test
using Random

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "patching" begin
    @testset "createpath" begin
        L = 3
        sitedims = [[2], [2], [2]]

        po = TCIA.PatchOrdering(collect(1:L))
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == Int[]
        @test TCIA.createpath(TCIA.Projector([[1], [1], [0]], sitedims), po) == [1, 1]

        po = TCIA.PatchOrdering(reverse(collect(1:L)))
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == Int[]
        @test TCIA.createpath(TCIA.Projector([[0], [0], [1]], sitedims), po) == [1]
    end

    @testset "tree" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]
        N = length(sitedims)
        bonddims = [1, 4, 4, 4, 1]
        po = TCIA.PatchOrdering(collect(1:N))
        tt = TCIA.ProjTensorTrain(
            TCI.TensorTrain([
                rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
            ]),
        )

        T = Float64
        NT = Union{TCIA.ProjTensorTrain{T},TCIA.LazyMatrixMul{T}}

        # Root node
        root = TCIA.create_node(NT, Int[])

        ptt = TCIA.project(tt, TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims))
        TCIA.add!(root, ptt, po)
        @test length(TCIA.find_node(root, Int[]).value) == 1
        @test TCIA.find_node(root, Int[]).value[1] == ptt

        ptt2 = TCIA.project(tt, TCIA.Projector([[0, 1], [0, 0], [0, 0], [0, 0]], sitedims))
        TCIA.add!(root, ptt2, po)
        @test length(TCIA.find_node(root, Int[]).value) == 2
        @test TCIA.find_node(root, Int[]).value[2] == ptt2

        ptt3 = TCIA.project(tt, TCIA.Projector([[1, 1], [1, 1], [0, 0], [0, 0]], sitedims))
        TCIA.add!(root, ptt3, po)
        @test length(TCIA.find_node(root, [1, 1]).value) == 1
        @test TCIA.find_node(root, [1, 1]).value[1] == ptt3
    end

    @testset "makechildproj" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]
        N = length(sitedims)

        let
            po = TCIA.PatchOrdering(collect(1:N))
            proj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
            @test TCIA.makechildproj(proj, po) == [
                TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 1], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 2], [0, 0], [0, 0], [0, 0]], sitedims),
            ]
        end

        let
            po = TCIA.PatchOrdering(reverse(collect(1:N)))
            proj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
            @test TCIA.makechildproj(proj, po) == [
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [1, 1]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [2, 1]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [1, 2]], sitedims),
                TCIA.Projector([[0, 0], [0, 0], [0, 0], [2, 2]], sitedims),
            ]
        end
    end

    @testset "makeproj" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]

        po = TCIA.PatchOrdering([1, 3, 2, 4])
        prefix = [[1, 2], [1, 1]]

        TCIA.makeproj(po, prefix, sitedims) ==
        TCIA.Projector([[1, 2], [0, 0], [1, 1], [0, 0]], sitedims)
    end
end
