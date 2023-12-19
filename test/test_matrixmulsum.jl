using Test
using LinearAlgebra
import TensorCrossInterpolation as TCI
import TCIAlgorithms: MatrixProduct
import TCIAlgorithms as TCIA

using ITensors

@testset "MatrixProductSum" begin
    N = 3
    bonddims_a = [1, 2, 3, 1]
    bonddims_b = [1, 2, 3, 1]
    @assert length(bonddims_a) == N + 1
    @assert length(bonddims_b) == N + 1

    localdims1 = [2, 2, 2]
    localdims2 = [3, 3, 3]
    localdims3 = [2, 2, 2]

    Nsum = 2

    atts = [
        TCI.TensorTrain([
            rand(bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n+1]) for n = 1:N
        ]) for _ = 1:Nsum
    ]
    btts = [
        TCI.TensorTrain([
            rand(bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n+1]) for n = 1:N
        ]) for _ = 1:Nsum
    ]

    products = [TCIA.MatrixProduct(a, b) for (a, b) in zip(atts, btts)]
    ab = TCIA.MatrixProductSum(products)

    @test TCI.isbatchevaluable(ab)

    @test ab([Int[]], [Int[]], Val(N)) ≈
          sum(p([Int[]], [Int[]], Val(N)) for p in products)
    @test ab(fill(1, N)) ≈ sum((p(fill(1, N)) for p in products))

end
