using Test
using LinearAlgebra
import TCIAlgorithms: contract, multiply, MPO, evaluate, batchevaluate

@testset "MPO-MPO contraction" begin
    N = 5
    A = MPO([rand(2, 3, 4, 2) for _ in 1:N])
    B = MPO([reshape(diagm([1, 1, 1]), (1, 3, 3, 1)) for _ in 1:N])
    C = MPO([reshape(diagm([1, 1, 1, 1]), (1, 4, 4, 1)) for _ in 1:N])

    R = contract(B, fill(2, N), A, fill(1, N))
    for (a, r) in zip(A, R)
        @test size(r) == (2, 3, 4, 2)
        @test a ≈ r
    end
    @test_throws DimensionMismatch contract(A, fill(2, N - 1), B, fill(1, N))
    @test_throws DimensionMismatch contract(A, fill(2, N), B, fill(1, N))

    @test all(contract(A, fill(2, N), C, fill(1, N)) .≈ A)
end

@testset "MPO-MPO deltaproduct" begin
    N = 5
    B = MPO([reshape(diagm([1, 1, 1]), (1, 3, 3, 1)) for _ in 1:N])
    B2 = multiply(B, [1, 2], B, [1, 2])
    for (b, b2) in zip(B, B2)
        @test b == b2
    end
    C = MPO([reshape([0 0 1; 0 1 0; 1 0 0], (1, 3, 3, 1)) for _ in 1:N])
    BC = multiply(B, [1, 2], C, [1, 2])
    for bc in BC
        @test bc == reshape(diagm([0, 1, 0]), (1, 3, 3, 1))
    end
end

@testset "MPO-MPO evaluate" begin
    N = 2
    A = MPO([rand(2, 3, 4, 2) for _ in 1:N])
    @test (@inferred evaluate(A, [[1, 1], [1, 1]]; usecache=true)) ==
          (@inferred evaluate(A, [[1, 1], [1, 1]]; usecache=false))
end

@testset "MPO-MPO batchevaluate" begin
    N = 4
    bonddims = fill(3, N + 1)
    bonddims[1] = 1
    bonddims[end] = 1
    A = MPO([rand(bonddims[n], 2, bonddims[n+1]) for n in 1:N])

    leftindexset = [[[1]], [[2]]]
    rightindexset = [[[1]], [[2]]]

    result = batchevaluate(A, leftindexset, rightindexset, Val(2))
    for cindex in [[[1], [1]], [[1], [2]]]
        for (il, lindex) in enumerate(leftindexset)
            for (ir, rindex) in enumerate(rightindexset)
                @test result[il, Iterators.flatten(cindex)..., ir] ≈ evaluate(A, vcat(lindex, cindex, rindex))
            end
        end
    end
end