using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "crossinterpolate" begin
    @testset "_FuncAdapterTCI2Subset" begin
        N = 6
        sitedims = [[2, 2] for _ in 1:N]
        bonddims = [1, fill(4, N - 1)..., 1]

        tt = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])

        p = TCIA.Projector([[0, 0], [0, 0], [2, 2], [0, 0], [0, 0], [1, 1]], sitedims)

        ptt = TCIA.project(TCIA.ProjTensorTrain(tt), p)

        ptt_wrapper = TCIA._FuncAdapterTCI2Subset(ptt)
        @test ptt_wrapper.localdims == [4, 4, 4, 4]
        @test length(ptt_wrapper) == 4

        @test ptt_wrapper([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈
            ptt([[1, 1], [1, 1], [2, 2], [1, 1], [1, 1], [1, 1]])
        @test ptt_wrapper([1, 1, 1, 1]) ≈ ptt([1, 1, 4, 1, 1, 1])

        leftindexset = [[1, 1]]

        rightindexset = [[1]]

        @test vec(ptt_wrapper(leftindexset, rightindexset, Val(1))) ≈
            vec([ptt([1, 1, 4, i, 1, 1]) for i in 1:4])
    end

    @testset "fulllength_leftindexset and fulllength_rightindexset" begin
        N = 5
        sitedims = [[2, 2] for _ in 1:N]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [0, 0]], sitedims)
        @test TCIA.fulllength_leftindexset(p, [Int[]]) == [[1]]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [0, 0]], sitedims)
        @test TCIA.fulllength_leftindexset(p, [[2]]) == [[1, 2]]

        p = TCIA.Projector([[0, 0], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightindexset(p, [Int[]]) == [[1]]

        p = TCIA.Projector([[0, 0], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightindexset(p, [[2]]) == [[2, 1]]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightindexset(p, [Int[]]) == [[1]]
    end

    @testset "fulllength_leftindexset (empty set)" begin
        projector = TCIA.Projector([[4], [4], [0]], [[4], [4], [4]])
        @test TCIA.fulllength_leftindexset(projector, [Int[]]) == [[4, 4]]
    end
end
