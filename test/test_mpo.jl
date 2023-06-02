using Test
using LinearAlgebra
import TCIAlgorithms: MPO, contract

@testset "MPO-MPO contraction" begin
    A = MPO([rand(2, 3, 4, 2) for _ in 1:5])
    B = MPO([reshape(diagm([1, 1, 1]), (1, 3, 3, 1)) for _ in 1:5])
    C = MPO([reshape(diagm([1, 1, 1, 1]), (1, 4, 4, 1)) for _ in 1:5])

    @test all(contract(B, fill(2, 5), A, fill(1, 5)) .≈ A)
    @test_throws DimensionMismatch contract(A, fill(2, 4), B, fill(1, 5))
    @test_throws DimensionMismatch contract(A, fill(2, 5), B, fill(1, 5))

    @test all(contract(A, fill(2, 5), C, fill(1, 5)) .≈ A)
end
