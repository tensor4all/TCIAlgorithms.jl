using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "mul" begin
    @testset "MPO-MPO contraction" begin
        N = 4
        bonddims_a = [1, 2, 3, 2, 1]
        bonddims_b = [1, 2, 3, 2, 1]
        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]
        localdims3 = [2, 2, 2, 2]

        a = TCI.TensorTrain{ComplexF64,4}([
            rand(
                ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1]
            ) for n in 1:N
        ])

        b = TCI.TensorTrain{ComplexF64,4}([
            rand(
                ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n + 1]
            ) for n in 1:N
        ])

        #ab_mul = TCIA.MatrixProduct(a, b)
        #ab = TCIA.contract(a, b; f=f)
        #@test TCI.sitedims(ab) == [[localdims1[i], localdims3[i]] for i in 1:N]
        #@test _tomat(ab) ≈ f.(_tomat(a) * _tomat(b))
    end
end