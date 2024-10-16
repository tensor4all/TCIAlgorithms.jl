using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "ProjTensorTrain" begin
    @testset "ProjTensorTrain" begin
        N = 4
        χ = 2
        bonddims = [1, χ, χ, χ, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
        localdims = collect(prod.(sitedims))

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])

        for (n, tensor) in enumerate(tt)
            size(tensor)[2:(end - 1)] == sitedims[n]
        end

        # Projection 
        prj = TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)
        _test_projection(TCIA.ProjTensorTrain(tt), prj)

        # Projection with truncation
        #globalprj = TCIA.Projector([[0, 0], [0, 0], [0, 0], [0, 0]], sitedims)
        ptt_truncated = TCIA.ProjTensorTrain{Float64}(tt)
        ptt_truncated = TCIA.project(ptt_truncated, prj; compression=true)
        indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
        @test tt(indexset1) ≈ ptt_truncated(indexset1)
    end

    @testset "ProjTensorTrain <= TensorTrain" begin
        N = 4
        bonddims = [1, 10, 10, 10, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [1, 1, 1, 1]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])
        prj = TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)
        ptt = TCIA.ProjTensorTrain(tt, prj)

        @test tt([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈ ptt([[1, 1], [1, 1], [1, 1], [1, 1]])
    end

    @testset "batchevaluateprj" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]

        N = length(sitedims)
        bonddims = [1, 4, 4, 4, 1]
        @assert length(bonddims) == N + 1

        tt = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])

        p = TCIA.Projector([[0, 0], [2, 2], [0, 0], [0, 0]], sitedims)

        ptt = TCIA.project(TCIA.ProjTensorTrain(tt), p; compression=true)

        leftindexset = [[[1, 1]]]
        rightmmultiidxset = [[[1, 1]]]
        batchprj = TCIA.batchevaluateprj(ptt, leftindexset, rightmmultiidxset, Val(2))

        @assert size(batchprj) == (1, 1, 4, 1)
        @test batchprj[1, 1, 1, 1] ≈ ptt([1, 4, 1, 1])
    end

    @testset "add" begin
        sitedims = [[2, 2], [2, 2], [2, 2], [2, 2]]

        N = length(sitedims)
        bonddims = [1, 4, 4, 4, 1]
        @assert length(bonddims) == N + 1

        proja = TCIA.Projector([[0, 0], [2, 2], [0, 0], [0, 0]], sitedims)
        projb = TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims)

        a = begin
            tt = TCI.TensorTrain([
                rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
            ])
            TCIA.project(TCIA.ProjTensorTrain(tt), proja; compression=false)
        end

        b = begin
            tt = TCI.TensorTrain([
                rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
            ])
            TCIA.project(TCIA.ProjTensorTrain(tt), projb; compression=false)
        end

        ab = TCIA.add(a, b; maxbonddim=2 * maximum(bonddims), tolerance=1e-14)

        @test TCIA.fulltensor(ab) ≈ TCIA.fulltensor(a) + TCIA.fulltensor(b)
    end

    @testset "project_on_subsetsiteinds" begin
        N = 4
        χ = 2
        bonddims = [1, χ, χ, χ, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        sitedims = [[x, y] for (x, y) in zip(localdims1, localdims2)]
        localdims = collect(prod.(sitedims))

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n + 1]) for n in 1:N
        ])

        for (n, tensor) in enumerate(tt)
            size(tensor)[2:(end - 1)] == sitedims[n]
        end

        # Projection 
        for prj in [
            TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedims),
            TCIA.Projector([[1, 1], [1, 1], [0, 0], [0, 0]], sitedims),
            TCIA.Projector([[0, 0], [1, 1], [0, 0], [0, 0]], sitedims),
            TCIA.Projector([[0, 0], [0, 0], [1, 1], [0, 0]], sitedims),
            TCIA.Projector([[0, 0], [0, 0], [1, 1], [1, 1]], sitedims),
            TCIA.Projector([[0, 0], [0, 0], [0, 0], [1, 1]], sitedims),
        ]
            prjtt = TCIA.project(TCIA.ProjTensorTrain(tt), prj)
            res = TCIA.project_on_subsetsiteinds(prjtt)
            localdims = Base.only.(TCI.sitedims(res))
            #@show localdims
            #@show size.(res.sitetensors)
            #@show size(TCIA.fulltensor(prjtt; reducesitedims=true))
            #@show size([res(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))])
            @test vec(TCIA.fulltensor(prjtt; reducesitedims=true)) ≈
                vec([res(collect(Tuple(i))) for i in CartesianIndices(Tuple(localdims))])
        end
    end
end
