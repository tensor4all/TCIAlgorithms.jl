using Test
using LinearAlgebra
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "ElementwiseProduct" for f in [x -> x, x -> 2 * x]
    N = 4
    bonddims = [1, 2, 3, 2, 1]
    @assert length(bonddims) == N + 1

    localdims = [2, 3, 3, 2]
    a = TCI.TensorTrain([rand(bonddims[n], localdims[n], bonddims[n + 1]) for n in 1:N])
    b = TCI.TensorTrain([rand(bonddims[n], localdims[n], bonddims[n + 1]) for n in 1:N])

    ab = TCIA.ElementwiseProduct([a, b]; f=f)

    @test TCI.isbatchevaluable(ab)

    tolerance = 1e-12
    tci, ranks, errors = TCI.crossinterpolate2(
        Float64, ab, localdims, [ones(Int, N)]; tolerance=tolerance
    )

    ab_res = Float64[]
    ab_ref = Float64[]
    for idx_ in Iterators.product((1:d for d in localdims)...)
        idx = collect(idx_)
        push!(ab_res, ab(idx))
        push!(ab_ref, f(a(idx) .* b(idx)))
    end

    @test ab_res ≈ ab_ref
end
