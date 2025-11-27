import QuanticsGrids as QG
using TensorCrossInterpolation
import TCIAlgorithms as TCIA
using HubbardAtoms
using SparseIR
using Quantics
using ITensors

@testset "BSE in 3D" begin
    U = 1.6
    beta = 2.3
    model = HubbardAtom(U, beta)
    R = 4
    N = 2^R
    maxbonddim = 40
    grid = QG.InherentDiscreteGrid{3}(
        R, (-N + 1, -N + 1, -N); step=2, unfoldingscheme=:fused
    )

    base = 2
    sitesk = [Index(base, "k=$n") for n in 1:R] # ν
    sitesk´ = prime.(sitesk) # ν´
    sitesq = [Index(base, "q=$n") for n in 1:R] # ω
    sites = collect(collect.(zip(sitesk, sitesk´, sitesq)))
    function makeverts(ch)
        function fq_full(x, y, z)
            return full_vertex(
                ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z))
            )
        end
        fI_full = x -> fq_full(quantics_to_origcoord(grid, x)...)

        # we absorb 1/β^2 into the chi0 function
        function fq_chi0(x, y, z)
            return 1 / beta^2 *
                   chi0(ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z)))
        end
        fI_chi0 = x -> fq_chi0(quantics_to_origcoord(grid, x)...)

        function fq_gamma(x, y, z)
            return gamma(ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z)))
        end
        fI_gamma = x -> fq_gamma(quantics_to_origcoord(grid, x)...)

        return fq_full, fq_chi0, fq_gamma, fI_full, fI_chi0, fI_gamma
    end
    function interpolateverts(fI_chi0, fI_full, fI_gamma)
        localdims = dim.(sites)
        sitedims = [dim.(s) for s in sites]
        pordering = TCIA.PatchOrdering(collect(1:R))
        initialpivots = [QG.origcoord_to_quantics(grid, 0)] # approx center of grid

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

        return full_patches, chi0_patches, gamma_patches
    end
    function makevertsdiagonal(full_patches, chi0_patches, gamma_patches)
        siteskk´_vec = [[x, y] for (x, y) in zip(sitesk, sitesk´)]
        sitesq_vec = [[z] for z in sitesq]
        sites_separateq = [x for pair in zip(siteskk´_vec, sitesq_vec) for x in pair]

        full_mps = TCIA.ProjMPSContainer(Float64, full_patches, sites)
        full_kk´_q = Quantics.rearrange_siteinds(full_mps, sites_separateq)
        full_kk´_qq´ = Quantics.makesitediagonal(full_kk´_q, "q")
        full_ptt = TCIA.ProjTTContainer{Float64}(full_kk´_qq´)

        chi0_mps = TCIA.ProjMPSContainer(Float64, chi0_patches, sites)
        chi0_kk´_q = Quantics.rearrange_siteinds(chi0_mps, sites_separateq)
        chi0_kk´_qq´ = Quantics.makesitediagonal(chi0_kk´_q, "q")
        chi0_kk´_q´q´´ = prime(chi0_kk´_qq´)
        chi0_ptt = TCIA.ProjTTContainer{Float64}(chi0_kk´_q´q´´)

        gamma_mps = TCIA.ProjMPSContainer(Float64, gamma_patches, sites)
        gamma_kk´_q = Quantics.rearrange_siteinds(gamma_mps, sites_separateq)
        gamma_kk´_qq´ = Quantics.makesitediagonal(gamma_kk´_q, "q")
        gamma_kk´_q´´q´´´ = prime(gamma_kk´_qq´, 2)
        gamma_ptt = TCIA.ProjTTContainer{Float64}(gamma_kk´_q´´q´´´)

        diagonal_sites = full_kk´_qq´.sites

        return full_ptt, chi0_ptt, gamma_ptt, diagonal_sites
    end
    function calculatebse(full_ptt, chi0_ptt, gamma_ptt, diagonal_sites)
        pordering = TCIA.PatchOrdering(collect(1:(2R)))

        chi0_gamma_ptt = TCIA.adaptivematmul(chi0_ptt, gamma_ptt, pordering; maxbonddim)
        phi_bse_diagonal = TCIA.adaptivematmul(
            full_ptt, chi0_gamma_ptt, pordering; maxbonddim
        )
        phi_bse_diagonal_projmps = TCIA.ProjMPSContainer(
            Float64, phi_bse_diagonal, diagonal_sites
        )
        phi_bse_projmps_kk´_q = Quantics.extractdiagonal(phi_bse_diagonal_projmps, "q")
        phi_bse_projmps_kk´q = Quantics.rearrange_siteinds(phi_bse_projmps_kk´_q, sites)
        phi_bse = TCIA.ProjTTContainer{Float64}(phi_bse_projmps_kk´q)

        return phi_bse
    end
    function comparereference(phi_bse, fq_full, fq_chi0, fq_gamma)
        # normal multiplication for comparison
        box = [
            (x, y, z) for x in range(-N + 1; step=2, length=N),
            y in range(-N + 1; step=2, length=N),
            z in range(-N; step=2, length=N)
        ]
        chi0_exact = map(splat(fq_chi0), box)
        full_exact = map(splat(fq_full), box)
        gamma_exact = map(splat(fq_gamma), box)
        phi_normalmul = stack(
            gamma_exact[:, :, i] * chi0_exact[:, :, i] * full_exact[:, :, i] for i in 1:N
        )

        phi_adaptivemul = [phi_bse(QG.origcoord_to_quantics(grid, p)) for p in box]

        return norm(phi_normalmul - phi_adaptivemul) / norm(phi_normalmul)
    end
    ch = DensityChannel()
    fq_full, fq_chi0, fq_gamma, fI_full, fI_chi0, fI_gamma = makeverts(ch)
    full_patches, chi0_patches, gamma_patches = interpolateverts(fI_chi0, fI_full, fI_gamma)
    full_ptt, chi0_ptt, gamma_ptt, diagonal_sites = makevertsdiagonal(
        full_patches, chi0_patches, gamma_patches
    )
    phi_bse = calculatebse(full_ptt, chi0_ptt, gamma_ptt, diagonal_sites)
    error = comparereference(phi_bse, fq_full, fq_chi0, fq_gamma)
    @test error < 7e-4
end
