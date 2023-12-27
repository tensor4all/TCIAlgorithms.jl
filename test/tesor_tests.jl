@testitem "test_tests.jl" begin

using Test
using LinearAlgebra
import TCIAlgorithms: contract, deltaproduct

@testset "Tensor contraction" begin
    A = rand(4, 5)
    B = rand(6, 5)

    @test contract(A, [2], B, [2]) == A * transpose(B)
    @test contract(A, 2, B, 2) == A * transpose(B)
    @test_throws DimensionMismatch contract(A, [2], B, [1])

    C = rand(3, 5, 2, 4)
    @test contract(C, [1], diagm([1, 1, 1]), [2]) ≈ permutedims(C, [2, 3, 4, 1])
    @test_throws DimensionMismatch contract(C, [2], diagm([1, 1, 1]), [2])
    @test contract(C, [4], diagm([0.5, 0.5, 0.5, 0.5]), [2]) == C ./ 2
end

@testset "Tensor elementwise multiplication" begin
    A = rand(4, 5)
    B = rand(4, 5)
    @test deltaproduct(A, [1, 2], B, [1, 2]) ≈ A .* B
    @test sum(deltaproduct(A, [1], B, [1]), dims = 2)[:, 1, :] ≈ transpose(A) * B
end

end

