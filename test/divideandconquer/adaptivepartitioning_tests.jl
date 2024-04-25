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

@testset "PatchOrdering" begin
    po = TCIA.PatchOrdering([4, 3, 2, 1])
    @test TCIA.maskactiveindices(po, 2) == [1, 1, 0, 0]
    @test TCIA.maskactiveindices(po, 1) == [1, 1, 1, 0]
    @test TCIA.fullindices(po, [[1]], [[2], [3], [4]]) == [[2], [3], [4], [1]]
    @test TCIA.fullindices(po, [[1], [2]], [[3], [4]]) == [[3], [4], [2], [1]]

    let
        po = TCIA.PatchOrdering([1, 3, 2, 4, 5])
        prefix = [[1], [2]]
        localdims = fill([10000], length(po))
        @test TCIA.Projector(po, prefix, localdims).data == [[1], [0], [2], [0], [0]]
    end
end

@testset "ProjectedTensorTrain from a tensor train on unprojected indices" begin
    T = Float64
    L = 5
    sitedims = [[2] for _ in 1:L]

    # projector = [1, 2, 0, 0, 0]
    prj = TCIA.Projector([[1], [2], [0], [0], [0]], sitedims)
    localdims = fill([2], L)

    χ = 3
    tt = TCI.TensorTrain{T,3}([rand(1, 2, χ), rand(χ, 2, χ), rand(χ, 2, 1)])

    ptt = TCIA.ProjectedTensorTrain(tt, localdims, prj)
    @test ptt([[1], [1], [1], [1], [1]]) == 0
    @test ptt([[1], [2], [1], [1], [1]]) ≈ tt([[1], [1], [1]])
end

@testset "2D fermi gk" begin
    @everywhere ek(kx, ky) = 2 * cos(kx) + 2 * cos(ky) - 1.0

    @everywhere function gk(kx, ky, β)
        iv = im * π / β
        return 1 / (iv - ek(kx, ky))
    end

    for _flipordering in [false, true]
        Random.seed!(1234)

        flipper = _flipordering ? x -> reverse(x) : x -> x

        R = 20
        grid = DiscretizedGrid{2}(R, (0.0, 0.0), (2π, 2π))
        localdims = fill(4, R)

        β = 20.0
        f = x -> gk(quantics_to_origcoord(grid, flipper(x))..., β)

        tol = 1e-5

        pordering = TCIA.PatchOrdering(flipper(collect(1:R)))

        creator = TCIA.TCI2PatchCreator(
            ComplexF64, f, localdims; maxbonddim=50, rtol=tol, verbosity=0, ntry=10
        )

        tree = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0, maxnleaves=1000)
        #@show collect(keys(tree))
        #@show length(tree)

        #_evaluate(x, idx) = FMPOC.evaluate(x, [[i] for i in idx])
        #for _ = 1:100
        #pivot = rand(1:4, R)
        #error_func = x -> abs(f(x) - _evaluate(tree, x))
        #pivot = TCI.optfirstpivot(error_func, localdims, pivot)
        #@test isapprox(_evaluate(tree, pivot), f(pivot); atol=10 * creator.atol)
        #end

    end
end

@everywhere fxy(x, y) = exp(-0.5 * (x^2 + y^2))

@testset "2D Gaussian" begin
    Random.seed!(1234)

    R = 40
    grid = DiscretizedGrid{2}(R, (-5, -5), (5, 5))
    localdims = fill(4, R)

    qf = x -> fxy(quantics_to_origcoord(grid, x)...)

    tol = 1e-7

    pordering = TCIA.PatchOrdering(collect(1:R))

    creator = TCIA.TCI2PatchCreator(
        Float64, qf, localdims; maxbonddim=40, rtol=tol, verbosity=0, ntry=10
    )

    partres = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0, maxnleaves=1000)

    sitedims = [[d] for d in localdims]
    partt = TCIA.PartitionedTensorTrain(partres, sitedims, pordering)
    qidx = origcoord_to_quantics(grid, (0.0, 1.0))

    @test isapprox(
        fxy(quantics_to_origcoord(grid, qidx)...), partt([[q] for q in qidx]); atol=tol
    )
end

@testset "zerofunction" begin
    Random.seed!(1234)

    R = 4
    localdims = fill(4, R)

    qf = x -> 0.0

    tol = 1e-7

    pordering = TCIA.PatchOrdering(collect(1:R))

    creator = TCIA.TCI2PatchCreator(
        Float64, qf, localdims; maxbonddim=10, rtol=tol, verbosity=0, ntry=10
    )

    partres = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0, maxnleaves=2)

    sitedims = [[d] for d in localdims]
    partt = TCIA.PartitionedTensorTrain(partres, sitedims, pordering)
    qidx = fill(1, R)
    @test partt([[q] for q in qidx]) == 0.0
end

@testset "2D Gaussian * 2D Gaussian" begin
    Random.seed!(1234)

    R = 20
    xmax = 10.0
    tol = 1e-10

    grid = DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax))
    localdims = fill(4, R)

    # Construct TCI for 2D Gaussian
    qf = x -> fxy(quantics_to_origcoord(grid, x)...)

    pordering = TCIA.PatchOrdering(collect(1:R))
    creator = TCIA.TCI2PatchCreator(
        Float64, qf, localdims; maxbonddim=40, rtol=tol, verbosity=0, ntry=10
    )

    partres = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0, maxnleaves=1000)

    sitedims = [[d] for d in localdims]
    partt = TCIA.PartitionedTensorTrain(partres, sitedims, pordering)
    qidx = origcoord_to_quantics(grid, (0.0, 1.0))

    @test isapprox(
        fxy(quantics_to_origcoord(grid, qidx)...), partt([[q] for q in qidx]); atol=tol
    )

    # 2D Gaussian * 2D Gaussian
    sitedimstto = [[2, 2] for _ in eachindex(localdims)]
    partto = reshape(partt, sitedimstto)
    partto.tensortrains[1].projector.data
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
end
