using Distributed
using Test
using Random

@everywhere using TensorCrossInterpolation
@everywhere import TensorCrossInterpolation as TCI
@everywhere import TCIAlgorithms as TCIA
@everywhere import QuanticsGrids:
    DiscretizedGrid, quantics_to_origcoord, origcoord_to_quantics
@everywhere import QuanticsGrids as QG


@testset "crossinterpolate_lazyeval" begin
    @testset "2D Gaussian * 2D Gaussian" begin
        Random.seed!(1234)

        R = 20
        xmax = 10.0
        grid = DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax))
        grid1 = DiscretizedGrid{1}(R, -xmax, xmax)
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]
        qf = x -> gaussian(quantics_to_origcoord(grid, x)...)
        tol = 1e-7

        pordering = TCIA.PatchOrdering(collect(1:R))
        exptt = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf, localdims), pordering; verbosity=0
            ),
            sitedims,
        )

        nested_quantics(x, y) = [
            collect(p) for
            p in zip(origcoord_to_quantics(grid1, x), origcoord_to_quantics(grid1, y))
        ]

        points = [(rand() * 10 - 5, rand() * 10 - 5) for i in 1:100]
        @test isapprox(
            [exptt(nested_quantics(p...)) for p in points],
            [qf(QG.origcoord_to_quantics(grid, p)) for p in points];
            atol=1e-5,
        )

        # Test lazy evaluation
        exp2(x, y) = sqrt(π) * exp(-0.5 * (x^2 + y^2))
        exp2lazy = TCIA.lazymatmul(exptt, exptt)

        @test isapprox(
            [exp2(p...) for p in points],
            (2xmax / 2^R) .* [exp2lazy(nested_quantics(p...)) for p in points],
            atol=1e-4,
        )

        exp2tt = TCIA.adaptiveinterpolate(exp2lazy, pordering; verbosity=0, maxbonddim=100)

        @test isapprox(
            [exp2(p...) for p in points],
            (2xmax / 2^R) .* [exp2tt(nested_quantics(p...)) for p in points],
            atol=1e-4,
        )
    end
end
