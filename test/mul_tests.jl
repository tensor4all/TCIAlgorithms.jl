using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul, makeprojectable

@testset "mul" begin
    @testset "lazymatmul" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1

        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]

        a = TCI.TensorTrain([
            rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
        ])
        b = TCI.TensorTrain([
            rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
        ])

        projector_a = Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedimsa)
        #pa = project(ProjTensorTrain(a), projector_a)
        pa = project(makeprojectable(a), projector_a)

        projector_b = Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedimsb)
        pb = project(makeprojectable(b), projector_b)

        ab = TCIA.lazymatmul(pa, pb)

        @test ab.sitedims == [[2, 3] for _ in 1:N]

        a_tt = TCI.TensorTrain{T,4}(ab.a.data, sitedimsa)
        b_tt = TCI.TensorTrain{T,4}(ab.b.data, sitedimsb)

        ab_ref = TCI.contract_naive(a_tt, b_tt)

        @test TCIA.fulltensor(ab) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))

        # Fit algorithm
        @test TCIA.fulltensor(TCIA.approxtt(ab)) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))
    end

    @testset "projecting lazymul" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1

        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]

        a = TCI.TensorTrain([
            rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
        ])
        b = TCI.TensorTrain([
            rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
        ])

        for p in [[[1, 1], [0, 0], [0, 0], [0, 0]], [[1, 0], [2, 0], [0, 0], [0, 0]]]
            ab = TCIA.lazymatmul(makeprojectable(a), makeprojectable(b))
            _test_projection(ab, Projector(p, sitedimsab))
        end
    end
end
