using Distributed
using Test
using Random

@everywhere using TensorCrossInterpolation
@everywhere import TensorCrossInterpolation as TCI
@everywhere import TCIAlgorithms as TCIA
@everywhere import QuanticsGrids:
    DiscretizedGrid, quantics_to_origcoord, origcoord_to_quantics
@everywhere import QuanticsGrids as QG

@testset "crossinterpolate" begin
    @testset "_FuncAdapterTCI2Subset" begin
        N = 6
        sitedims = [[2, 2] for _ in 1:N]
        bonddims = [1, fill(4, N - 1)..., 1]

        tt = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])

        p = TCIA.Projector([[0, 0], [0, 0], [2, 2], [0, 0], [0, 0], [1, 1]], sitedims)

        ptt = TCIA.project(TCIA.ProjTensorTrain(tt), p)

        ptt_wrapper = TCIA._FuncAdapterTCI2Subset(ptt)
        @test ptt_wrapper.localdims == [4, 4, 4, 4]
        @test length(ptt_wrapper) == 4

        @test ptt_wrapper([[1, 1], [1, 1], [1, 1], [1, 1]]) ≈
            ptt([[1, 1], [1, 1], [2, 2], [1, 1], [1, 1], [1, 1]])
        @test ptt_wrapper([1, 1, 1, 1]) ≈ ptt([1, 1, 4, 1, 1, 1])

        leftindexset = [[1, 1]]

        rightmmultiidxset = [[1]]

        @test vec(ptt_wrapper(leftindexset, rightmmultiidxset, Val(1))) ≈
            vec([ptt([1, 1, 4, i, 1, 1]) for i in 1:4])
    end

    @testset "fulllength_leftmmultiidxset and fulllength_rightmmultiidxset" begin
        N = 5
        sitedims = [[2, 2] for _ in 1:N]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [0, 0]], sitedims)
        @test TCIA.fulllength_leftmmultiidxset(p, [Int[]]) == [[1]]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [0, 0]], sitedims)
        @test TCIA.fulllength_leftmmultiidxset(p, [[2]]) == [[1, 2]]

        p = TCIA.Projector([[0, 0], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightmmultiidxset(p, [Int[]]) == [[1]]

        p = TCIA.Projector([[0, 0], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightmmultiidxset(p, [[2]]) == [[2, 1]]

        p = TCIA.Projector([[1, 1], [0, 0], [1, 1], [0, 0], [1, 1]], sitedims)
        @test TCIA.fulllength_rightmmultiidxset(p, [Int[]]) == [[1]]
    end

    @testset "fulllength_leftmmultiidxset (empty set)" begin
        projector = TCIA.Projector([[4], [4], [0]], [[4], [4], [4]])
        @test TCIA.fulllength_leftmmultiidxset(projector, [Int[]]) == [[4, 4]]
    end

    @testset "2D Guassian" begin
        Random.seed!(1234)

        R = 40
        grid = DiscretizedGrid{2}(R, (-5, -5), (5, 5))
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]

        qf = x -> gaussian(quantics_to_origcoord(grid, x)...)

        tol = 1e-7

        pordering = TCIA.PatchOrdering(collect(1:R))

        creator = TCIA.TCI2PatchCreator(
            Float64,
            TCIA.makeprojectable(Float64, qf, localdims),
            localdims,
            ;
            maxbonddim=20,
            tolerance=tol,
            verbosity=0,
            ntry=10,
        )

        obj = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0)

        points = [(rand() * 10 - 5, rand() * 10 - 5) for i in 1:100]

        @test isapprox(
            [obj(QG.origcoord_to_quantics(grid, p)) for p in points],
            [qf(QG.origcoord_to_quantics(grid, p)) for p in points];
            atol=1e-5,
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
            Float64, qf, localdims; maxbonddim=10, tolerance=tol, verbosity=0, ntry=10
        )

        obj = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0)

        qidx = fill(1, R)
        @test obj([[q] for q in qidx]) == 0.0
    end

    @testset "approxtt" begin
        N = 6
        sitedims = [[4] for _ in 1:N]
        bonddims = [1, fill(10, N - 1)..., 1]

        localdims = fill(4, N)

        tol = 1e-7

        tt = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])

        p = TCIA.Projector([[0], [0], [0], [0], [0], [0]], sitedims)
        pordering = TCIA.PatchOrdering(collect(1:N))

        ptt = TCIA.project(TCIA.ProjTensorTrain(tt), p)

        obj = TCIA.adaptiveinterpolate(ptt; verbosity=0, maxbonddim=5)

        @test vec(TCIA.fulltensor(obj)) ≈ vec(TCIA.fulltensor(ptt))
    end

    @testset "randompatchorder&2Dgreen" begin
        Random.seed!(1234)

        function green(kx, ky; ω=0.1, δ=0.01)
            denominator = ω + 2 * cos(kx) + 2 * cos(ky) + im * δ
            return real(1 / denominator)
        end

        tol = 1e-5
        mb = 30
        R = 8

        grid = QG.DiscretizedGrid{2}(R, (-pi, -pi), (pi, pi))
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]

        qf = x -> green(QG.quantics_to_origcoord(grid, x)...)

        pordering = TCIA.PatchOrdering(shuffle(collect(1:R)))

        creator = TCIA.TCI2PatchCreator(
            Float64,
            TCIA.makeprojectable(Float64, qf, localdims),
            localdims,
            ;
            maxbonddim=mb,
            tolerance=tol,
            verbosity=0,
            ntry=10,
        )
        obj = TCIA.adaptiveinterpolate(creator, pordering; verbosity=0)

        points = [(rand() * 2 * pi - pi, rand() * 2 * pi - pi) for i in 1:1000]

        @assert length(obj) > localdims[1]

        @test isapprox(
            [obj(QG.origcoord_to_quantics(grid, p)) for p in points],
            [qf(QG.origcoord_to_quantics(grid, p)) for p in points];
            atol=1e-4,
        )
    end

    @testset "recyclepivots" begin
        Random.seed!(1234)

        function linear_gaussians(
            x::Float64, y::Float64; centers::Vector{Vector{Float64}}=[[0.0, 0.0]]
        )
            N_gauss = length(centers)
            input_vec = [x, y]
            σ = [2.0^(-j - 1) for j in 1:N_gauss]
            return sum(exp(-norm(input_vec - centers[j])^2 / σ[j]^2) for j in 1:N_gauss)
        end

        R = 30
        grid = QG.DiscretizedGrid{2}(R, (-3.0, -3.0), (3.0, 3.0))
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]

        gauss_centers = [[-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0], [2.0, 2.0]]
        qf =
            x -> linear_gaussians(
                QG.quantics_to_origcoord(grid, x)...; centers=gauss_centers
            )

        pordering = TCIA.PatchOrdering(collect(1:R))
        tol = 1e-7
        mb = 20

        # Provide very good pivots on the maximums of the function 
        initialpivots = [
            TCI.optfirstpivot(qf, localdims, QG.origcoord_to_quantics(grid, Tuple(c))) for
            c in gauss_centers
        ]

        creator_recycle = TCIA.TCI2PatchCreator(
            Float64,
            TCIA.makeprojectable(Float64, qf, localdims),
            localdims,
            ;
            maxbonddim=mb,
            tolerance=tol,
            recyclepivots=true,
            initialpivots=initialpivots,
            ninitialpivot=0,
        )

        creator_no_recycle = TCIA.TCI2PatchCreator(
            Float64,
            TCIA.makeprojectable(Float64, qf, localdims),
            localdims,
            ;
            maxbonddim=mb,
            tolerance=tol,
            recyclepivots=false,
            initialpivots=initialpivots,
            ninitialpivot=0,
        )

        # Perform ony one patch iteration
        _, recycle_patches = TCIA.__taskfunc(creator_recycle, pordering)
        _, no_recycle_patches = TCIA.__taskfunc(creator_no_recycle, pordering)

        # Check if the "good" pivots are conserved in each patch
        for patch in recycle_patches
            prj_val = collect(Iterators.flatten(patch.projector))[1]
            patch_pivots = vcat.([prj_val], patch.initialpivots)

            @test intersect(patch_pivots, initialpivots) ==
                filter(piv -> piv[1] == prj_val, initialpivots)
        end

        # Check that the information is lost
        for patch in no_recycle_patches
            prj_val = collect(Iterators.flatten(patch.projector))[1]
            patch_pivots = vcat.([prj_val], patch.initialpivots)

            @test intersect(patch_pivots, initialpivots) !==
                filter(piv -> piv[1] == prj_val, initialpivots)
        end
    end
end
