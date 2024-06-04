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
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == [0, 0, 0]
        @test TCIA.createpath(TCIA.Projector([[0], [1], [0]], sitedims), po) == [0, 1, 0]

        po = TCIA.PatchOrdering(reverse(collect(1:L)))
        @test TCIA.createpath(TCIA.Projector([[0], [0], [0]], sitedims), po) == [0, 0, 0]
        @test TCIA.createpath(TCIA.Projector([[0], [0], [1]], sitedims), po) == [1, 0, 0]
        @test TCIA.createpath(TCIA.Projector([[0], [2], [1]], sitedims), po) == [1, 2, 0]
    end

    @testset "tree" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]
        N = length(sitedims)
        bonddims = [1, 4, 4, 4, 1]
        po = TCIA.PatchOrdering(collect(1:N))
        tt = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])

        T = Float64
        NT = Union{TCIA.ProjTensorTrain{T},TCIA.LazyMatrixMul{T}}

        # Root node
        root = TCIA.create_node(NT, [0, 0, 0, 0], nothing)

        ptt = TCIA.project(tt, TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims))
        TCIA.add_node!(root, ptt)
    end
end
