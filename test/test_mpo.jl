using Test
using LinearAlgebra
import TCIAlgorithms: contract, multiply, MPO

@testset "MPO-MPO contraction" begin
    A = MPO([rand(2, 3, 4, 2) for _ in 1:5])
    B = MPO([reshape(diagm([1, 1, 1]), (1, 3, 3, 1)) for _ in 1:5])
    C = MPO([reshape(diagm([1, 1, 1, 1]), (1, 4, 4, 1)) for _ in 1:5])

    R = contract(B, fill(2, 5), A, fill(1, 5))
    for (a, r) in zip(A, R)
        @test size(r) == (2, 3, 4, 2)
        @test a ≈ r
    end
    @test_throws DimensionMismatch contract(A, fill(2, 4), B, fill(1, 5))
    @test_throws DimensionMismatch contract(A, fill(2, 5), B, fill(1, 5))

    @test all(contract(A, fill(2, 5), C, fill(1, 5)) .≈ A)
end

@testset "MPO-MPO deltaproduct" begin
    B = MPO([reshape(diagm([1, 1, 1]), (1, 3, 3, 1)) for _ in 1:5])
    B2 = multiply(B, [1, 2], B, [1, 2])
    for (b, b2) in zip(B, B2)
        @test b == b2
    end
    C = MPO([reshape([0 0 1; 0 1 0; 1 0 0], (1, 3, 3, 1)) for _ in 1:5])
    BC = multiply(B, [1, 2], C, [1, 2])
    for bc in BC
        @test bc == reshape(diagm([0, 1, 0]), (1, 3, 3, 1))
    end
end
