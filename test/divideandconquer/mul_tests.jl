using Distributed
using Test
using Random

# Define the maximum number of worker processes.
const MAX_WORKERS = 4

# Add worker processes if necessary.
addprocs(max(0, MAX_WORKERS - nworkers()))

@everywhere using TensorCrossInterpolation
@everywhere import TensorCrossInterpolation as TCI
@everywhere import TCIAlgorithms as TCIA
@everywhere using ITensors
@everywhere ITensors.disable_warn_order()
@everywhere import QuanticsGrids:
    DiscretizedGrid, quantics_to_origcoord, origcoord_to_quantics
@everywhere import QuanticsGrids as QG

@everywhere fxy2(x, y) = exp(-0.5 * (x^2 + y^2))

@testset "2D Gaussian * 2D Gaussian" begin

    Random.seed!(1234)

    R = 20
    xmax = 10.0
    tol = 1e-10

    grid = DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax))
    localdims = fill(4, R)

    # Construct TCI for 2D Gaussian
    qf = x -> fxy2(quantics_to_origcoord(grid, x)...)

    pordering = TCIA.PatchOrdering(collect(1:R))
    creator = TCIA.TCI2PatchCreator(
        Float64, qf, localdims; maxbonddim=40, rtol=tol, verbosity=0, ntry=10
    )

    partres = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0, maxnleaves=1000)

    sitedims = [[d] for d in localdims]
    partt = TCIA.PartitionedTensorTrain(partres, sitedims, pordering)
    qidx = origcoord_to_quantics(grid, (0.0, 1.0))

    @test isapprox(
        fxy2(quantics_to_origcoord(grid, qidx)...), partt([[q] for q in qidx]); atol=tol
    )

    # 2D Gaussian * 2D Gaussian
    sitedimstto = [[2, 2] for _ in eachindex(localdims)]
    partto = reshape(partt, sitedimstto)

    ttoprod = TCIA.create_multiplier(partto, partto)

    grid1 = DiscretizedGrid{1}(R, -xmax, xmax)
    ff(x, y) = sqrt(π) * exp(-0.5 * (x^2 + y^2))
    @test ff(0.0, 0.0) ≈
        (2xmax / 2^R) * ttoprod([[x, x] for x in origcoord_to_quantics(grid1, 0.0)])

    # TCI of 2D Gaussian * 2D Gaussian
    creator2 = TCIA.TCI2PatchCreator(
        Float64, ttoprod, localdims; maxbonddim=40, rtol=tol, verbosity=0, ntry=10
    )
    partres2 = TCIA.adaptiveinterpolate(creator2, pordering; verbosity=0, maxnleaves=1000)

    partt2 = reshape(
        TCIA.PartitionedTensorTrain(partres2, fill([4], R), pordering), fill([2, 2], R)
    )

    @test ff(0.0, 0.0) ≈
        (2xmax / 2^R) * partt2([[x, x] for x in origcoord_to_quantics(grid1, 0.0)])

    # Test contract()
    partt2_bycontract = reshape(
        TCIA.contract_tto(partto, partto; maxbonddim=40, rtol=tol, verbosity=0, ntry=10),
        fill([2, 2], R)
    )

    @test ff(0.0, 0.0) ≈
        (2xmax / 2^R) * partt2_bycontract([[x, x] for x in origcoord_to_quantics(grid1, 0.0)])
end
