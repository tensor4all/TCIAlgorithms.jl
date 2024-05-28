using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul

@testset "lazymatmul" begin
    T = Float64
    N = 4
    bonddims = [1, 3, 3, 3, 1]
    @assert length(bonddims) == N + 1

    sitedimsa = [[2, 2] for _ in 1:N]
    sitedimsb = [[2, 3] for _ in 1:N]
    sitedimsab = [[2, 3] for _ in 1:N]

    a = TCI.TensorTrain([rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N])
    b = TCI.TensorTrain([rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N])

    projector_a = Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedimsa)
    pa = project(ProjTensorTrain(a), projector_a)

    projector_b = Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedimsb)
    pb = project(ProjTensorTrain(b), projector_b)

    ab = TCIA.lazymatmul(pa, pb)

    @test ab.sitedims == [[2, 3] for _ in 1:N]

    indexset = [[1, 2], [1, 1], [1, 1], [1, 1]]
    a_tt = TCI.TensorTrain{T,4}(ab.a.data, sitedimsa)
    b_tt = TCI.TensorTrain{T,4}(ab.b.data, sitedimsb)
    @test ab(indexset) ≈ TCI.Contraction(a_tt, b_tt)(TCIA.lineari(sitedimsab, indexset))

    ab_ref = TCI.contract_naive(a_tt, b_tt)

    @test TCIA.fulltensor(ab) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))
end
