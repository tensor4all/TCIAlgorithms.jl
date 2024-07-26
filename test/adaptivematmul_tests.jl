using Test
using Random

import QuanticsGrids as QG
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using HubbardAtoms
using SparseIR

import TCIAlgorithms:
    create_node,
    add!,
    find_node,
    all_nodes,
    delete!,
    ProjTensorTrain,
    Projector,
    project,
    ProjTTContainer,
    adaptivematmul,
    BlockStructure,
    matmul

@testset "adaptivematmul" begin
    @testset "adpativematmul" begin
        Random.seed!(1234)
        T = Float64
        N = 3
        bonddims = [1, 10, 10, 1]
        @assert length(bonddims) == N + 1

        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]

        a = ProjTensorTrain(
            TCI.TensorTrain([
                rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
            ]),
        )
        b = ProjTensorTrain(
            TCI.TensorTrain([
                rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
            ]),
        )

        pa = ProjTTContainer([
            project(a, p) for p in [
                TCIA.Projector([[1, 1], [0, 0], [0, 0]], sitedimsa),
                TCIA.Projector([[2, 2], [0, 0], [0, 0]], sitedimsa),
            ]
        ])

        pb = ProjTTContainer([
            project(b, p) for p in [
                TCIA.Projector([[1, 1], [0, 0], [0, 0]], sitedimsb),
                TCIA.Projector([[2, 2], [0, 0], [0, 0]], sitedimsb),
            ]
        ])

        pordering = TCIA.PatchOrdering(collect(1:N))

        ab = adaptivematmul(pa, pb, pordering; maxbonddim=4)

        amat = reshape(permutedims(TCIA.fulltensor(pa), (1, 3, 5, 2, 4, 6)), 2^3, 2^3)
        bmat = reshape(permutedims(TCIA.fulltensor(pb), (1, 3, 5, 2, 4, 6)), 2^3, 3^3)

        abmat = reshape(permutedims(TCIA.fulltensor(ab), (1, 3, 5, 2, 4, 6)), 2^3, 3^3)
        @test abmat ≈ amat * bmat
    end

    @testset "mergesmalpacthes" begin
        Random.seed!(1234)
        T = Float64
        N = 4
        χ = 10
        bonddims = [1, fill(χ, N - 1)..., 1]
        tolerance = 1e-8
        @assert length(bonddims) == N + 1

        sitedims = [[2, 2] for _ in 1:N]

        a = ProjTensorTrain(
            TCI.TensorTrain([
                randn(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
            ]),
        )

        projectors = Projector[]
        rest = [[0, 0] for _ in 1:(N - 2)]
        for i1 in 1:2, j1 in 1:2, i2 in 1:2, j2 in 1:2
            push!(
                projectors,
                TCIA.Projector([[i1, j1], [i2, j2], deepcopy(rest)...], sitedims),
            )
        end

        pa = ProjTTContainer([
            project(a, p; compression=true, tolerance=tolerance, maxbonddim=1) for
            p in projectors
        ])

        @test length(pa.data) == 16

        pordering = TCIA.PatchOrdering(collect(1:N))
        root = TCIA.create_node(ProjTensorTrain{T}, Int[])
        for x in pa
            TCIA.add!(root, x, pordering)
        end

        maxbonddim = 10
        results = TCIA._mergesmallpatches(root; tolerance, maxbonddim=maxbonddim)

        @test 1 < length(results) < 16 # This is not an actual test, just a sanity check

        ref = TCIA.fulltensor(pa)
        reconst = TCIA.fulltensor(TCIA.ProjTTContainer(results))

        @test ref ≈ reconst
    end

    @testset "matmul with blocks" begin
        T = Float64
        sitedims = [[2, 2], [2, 2], [2, 2]]
        N = length(sitedims)
        bonddims = [1, 3, 3, 1]

        _random_tt() = TCI.TensorTrain([
            rand(bonddims[n], sitedims[n]..., bonddims[n + 1]) for n in 1:N
        ])
        _random_tt(bs::BlockStructure) =
            [project(ProjTensorTrain(_random_tt()), p) for p in bs]

        bs = TCIA.BlockStructure(
            vec([TCIA.Projector([[i, j], [0, 0], [0, 0]], sitedims) for i in 1:2, j in 1:2])
        )

        a = ProjTTContainer(_random_tt(bs))
        b = ProjTTContainer(_random_tt(bs))
        ab = TCIA.matmul(a, b, bs)

        ab_ref = TCI.contract_naive(
            TCI.TensorTrain{T,4}(TCIA.approxtt(a).data, sitedims),
            TCI.TensorTrain{T,4}(TCIA.approxtt(b).data, sitedims),
        )
        @test TCIA.fulltensor(TCIA.approxtt(ab)) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))
    end

    @testset "2d Gaussians" begin
        Random.seed!(1234)
        gaussian(x, y) = exp(-0.5 * (x^2 + y^2))
        R = 20
        xmax = 10.0
        grid = QG.DiscretizedGrid{2}(R, (-xmax, -xmax), (xmax, xmax))
        grid1 = QG.DiscretizedGrid{1}(R, -xmax, xmax)
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]
        qf = x -> gaussian(QG.quantics_to_origcoord(grid, x)...)
        pordering = TCIA.PatchOrdering(collect(1:R))

        expttpatches = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf, localdims),
                pordering;
                verbosity=0,
                maxbonddim=30,
            ),
            sitedims,
        )

        product = TCIA.adaptivematmul(expttpatches, expttpatches, pordering; maxbonddim=50)

        nested_quantics(x, y) = [
            collect(p) for p in
            zip(QG.origcoord_to_quantics(grid1, x), QG.origcoord_to_quantics(grid1, y))
        ]

        points = [(rand() * 10 - 5, rand() * 10 - 5) for i in 1:100]
        expproduct(x, y) = sqrt(π) * exp(-0.5 * (x^2 + y^2))

        @test isapprox(
            [expproduct(p...) for p in points],
            (2xmax / 2^R) .* [product(nested_quantics(p...)) for p in points], #(2xmax/2^R) = Δx, Δy
            atol=1e-3,
        )
    end

    @testset "polynomial integral" begin
        Random.seed!(1234)
        # \int_0^1 dz f1(x,z)*f2(z,y)
        f1(x, y) = x * y
        f2(x, y) = (x * y)^2
        R = 20
        grid = QG.DiscretizedGrid{2}(R, (0, 0), (1, 1))
        grid1 = QG.DiscretizedGrid{1}(R, 0, 1)
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]
        qf1 = x -> f1(QG.quantics_to_origcoord(grid, x)...)
        qf2 = x -> f2(QG.quantics_to_origcoord(grid, x)...)

        pordering = TCIA.PatchOrdering(collect(1:R))

        tt_f1 = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf1, localdims), pordering; verbosity=0
            ),
            sitedims,
        )

        tt_f2 = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf2, localdims), pordering; verbosity=0
            ),
            sitedims,
        )

        tt_f1_patches = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf1, localdims),
                pordering;
                verbosity=0,
                maxbonddim=8,
            ),
            sitedims,
        )

        tt_f2_patches = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf2, localdims),
                pordering;
                verbosity=0,
                maxbonddim=8,
            ),
            sitedims,
        )

        product_without_patches = TCIA.adaptivematmul(tt_f1, tt_f2, pordering)
        product = TCIA.adaptivematmul(
            tt_f1_patches, tt_f2_patches, pordering; maxbonddim=20
        )

        nested_quantics(x, y) = [
            collect(p) for p in
            zip(QG.origcoord_to_quantics(grid1, x), QG.origcoord_to_quantics(grid1, y))
        ]

        points = [(rand(), rand()) for i in 1:100]
        exact_product(x, y) = x * y^2 / 4 #integrated (xy)*(yz)^2 dz from 0 to 1

        #test without patches
        @test isapprox(
            [exact_product(p...) for p in points],
            (1 / 2^R) .* [product_without_patches(nested_quantics(p...)) for p in points],
            atol=1e-4,
        )

        #test with patches
        @test isapprox(
            [exact_product(p...) for p in points],
            (1 / 2^R) .* [product(nested_quantics(p...)) for p in points],
            atol=1e-4,
        )
    end

    @testset "diagonal matrices" begin
        Random.seed!(1234)
        f1(x, y) = ==(x, y) * x^2 # diagonal matrix with x^2 in diagonal
        f2(x, y) = ==(x, y) * (x^3) # diagonal matrix with x^3 in diagonal
        R = 5
        grid = QG.InherentDiscreteGrid{2}(R, (0, 0); step=(1, 1)) # from 0 to 2^R-1 = 31
        grid1 = QG.InherentDiscreteGrid{1}(R, 0; step=1)
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]
        qf1 = x -> f1(QG.quantics_to_origcoord(grid, x)...)
        qf2 = x -> f2(QG.quantics_to_origcoord(grid, x)...)
        initialpivots = [QG.origcoord_to_quantics(grid, (2^R - 1, 2^R - 1))] #largest element
        pordering = TCIA.PatchOrdering(collect(1:R))

        tt_f1 = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf1, localdims),
                pordering;
                initialpivots=initialpivots,
                verbosity=0,
            ),
            sitedims,
        )

        tt_f2 = reshape(
            TCIA.adaptiveinterpolate(
                TCIA.makeprojectable(Float64, qf2, localdims),
                pordering;
                initialpivots=initialpivots,
                verbosity=0,
            ),
            sitedims,
        )

        tt_f1_projected = TCIA.ProjTTContainer([
            TCIA.project(tt_f1[1], p) for p in [
                TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 2], [0, 0], [0, 0], [0, 0], [0, 0]], sitedims),
            ]
        ])

        tt_f2_projected = TCIA.ProjTTContainer([
            TCIA.project(tt_f2[1], p) for p in [
                TCIA.Projector([[1, 1], [0, 0], [0, 0], [0, 0], [0, 0]], sitedims),
                TCIA.Projector([[2, 2], [0, 0], [0, 0], [0, 0], [0, 0]], sitedims),
            ]
        ])

        product = TCIA.adaptivematmul(
            tt_f1_projected, tt_f2_projected, pordering; maxbonddim=5
        )
        product_without_patches = TCIA.adaptivematmul(
            tt_f1_projected, tt_f2_projected, pordering; maxbonddim=10
        )

        nested_quantics(x, y) = [
            collect(p) for p in
            zip(QG.origcoord_to_quantics(grid1, x), QG.origcoord_to_quantics(grid1, y))
        ]

        C = zeros(2^R, 2^R) .+ 0.0
        for i in 0:(2^R - 1)
            C[i + 1, i + 1] = i^5
        end

        product_matrix = zeros(2^R, 2^R) .+ 0.0
        product_matrix_without_patches = zeros(2^R, 2^R) .+ 0.0
        for i in 0:(2^R - 1), j in 0:(2^R - 1)
            product_matrix[i + 1, j + 1] = product(nested_quantics(i, j))
            product_matrix_without_patches[i + 1, j + 1] = product_without_patches(
                nested_quantics(i, j)
            )
        end

        @test maximum(abs.(product_matrix .- C)) < 1e-5
        @test maximum(abs.(product_matrix_without_patches .- C)) < 1e-5
    end

    @testset "Bethe-Salpeter equation" begin
        U = 1.6
        beta = 2.3
        model = HubbardAtom(U, beta)
        ch_d = DensityChannel()
        ch_m = MagneticChannel()
        ch_t = TripletChannel()
        ch_s = SingletChannel()
        m = BosonicFreq(10)

        R = 7
        maxbonddim = 50
        N = 2^R
        halfN = 2^(R - 1)
        grid = QG.InherentDiscreteGrid{2}(
            R, (-halfN, -halfN); step=(1, 1), unfoldingscheme=:fused
        )
        localdims = fill(4, R)
        sitedims = [[2, 2] for _ in 1:R]
        pordering = TCIA.PatchOrdering(collect(1:R))

        for ch in CHANNELS

            ######################### quantics functions ############################
            # absorb 1/β^2 into chi0 function!!!!!
            function fq_chi0(x, y)
                return 1 / beta^2 *
                       chi0(ch, model, (FermionicFreq(2x + 1), FermionicFreq(2y + 1), m))
            end
            fI_chi0 = QG.quanticsfunction(ComplexF64, grid, fq_chi0)

            function fq_full(x, y)
                return full_vertex(
                    ch, model, (FermionicFreq(2x + 1), FermionicFreq(2y + 1), m)
                )
            end
            fI_full = QG.quanticsfunction(ComplexF64, grid, fq_full)

            function fq_gamma(x, y)
                return gamma(ch, model, (FermionicFreq(2x + 1), FermionicFreq(2y + 1), m))
            end
            fI_gamma = QG.quanticsfunction(ComplexF64, grid, fq_gamma)
            #########################################################################

            initialpivots = [QG.origcoord_to_quantics(grid, (0, 0))] # approx center of grid

            chi0_patches = reshape(
                TCIA.adaptiveinterpolate(
                    TCIA.makeprojectable(Float64, fI_chi0, localdims),
                    pordering;
                    verbosity=0,
                    maxbonddim,
                    initialpivots,
                ),
                sitedims,
            )
            full_patches = reshape(
                TCIA.adaptiveinterpolate(
                    TCIA.makeprojectable(Float64, fI_full, localdims),
                    pordering;
                    verbosity=0,
                    maxbonddim,
                    initialpivots,
                ),
                sitedims,
            )
            gamma_patches = reshape(
                TCIA.adaptiveinterpolate(
                    TCIA.makeprojectable(Float64, fI_gamma, localdims),
                    pordering;
                    verbosity=0,
                    maxbonddim,
                    initialpivots,
                ),
                sitedims,
            )

            # multiplication Φ = Γ X₀ F
            chi0_full = TCIA.adaptivematmul(
                chi0_patches, full_patches, pordering; maxbonddim
            )
            phi_bse = TCIA.adaptivematmul(gamma_patches, chi0_full, pordering; maxbonddim)

            # normal multiplication for comparison
            box = [(x, y) for x in (-halfN):(halfN - 1), y in (-halfN):(halfN - 1)]
            chi0_exact = map(splat(fq_chi0), box)
            full_exact = map(splat(fq_full), box)
            gamma_exact = map(splat(fq_gamma), box)
            phi_normalmul = gamma_exact * chi0_exact * full_exact

            phi_adaptivemul = [phi_bse(QG.origcoord_to_quantics(grid, p)) for p in box]

            @test isapprox(phi_normalmul, phi_adaptivemul; rtol=1e-5)
        end
    end
end
