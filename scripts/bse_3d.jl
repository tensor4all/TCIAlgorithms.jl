import QuanticsGrids as QG
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using HubbardAtoms
using SparseIR
import Quantics: rearrange_siteinds
using ITensors

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

##

U = 1.6
beta = 2.3
model = HubbardAtom(U, beta)
ch_d = DensityChannel()
ch_m = MagneticChannel()
ch_t = TripletChannel()
ch_s = SingletChannel()
m = BosonicFreq(10)

R = 4
N = 2^R
maxbonddim = 40
grid = QG.InherentDiscreteGrid{3}(R, (-N + 1, -N + 1, -N); step=2, unfoldingscheme=:fused)

base = 2
sitesx = [Index(base, "x=$n") for n in 1:R] # ν
sitesy = [Index(base, "y=$n") for n in 1:R] # ν´
sitesz = [Index(base, "z=$n") for n in 1:R] # ω
sites = collect(collect.(zip(sitesx, sitesy, sitesz)))

localdims = dim.(sites)
sitedims = [dim.(s) for s in sites]

pordering = TCIA.PatchOrdering(collect(1:R))

# for ch in CHANNELS
ch = ch_d

######################### quantics functions ############################
# absorb 1/β^2 into chi0 function!!!!!
function fq_chi0(x, y, z)
    return 1 / beta^2 *
           chi0(ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z)))
end
fI_chi0 = QG.quanticsfunction(ComplexF64, grid, fq_chi0)

function fq_full(x, y, z)
    return full_vertex(ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z)))
end
fI_full = QG.quanticsfunction(ComplexF64, grid, fq_full)

function fq_gamma(x, y, z)
    return gamma(ch, model, (FermionicFreq(x), FermionicFreq(y), BosonicFreq(z)))
end
fI_gamma = QG.quanticsfunction(ComplexF64, grid, fq_gamma)
#########################################################################

initialpivots = [QG.origcoord_to_quantics(grid, 0)] # approx center of grid

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

##

chi0_mps = TCIA.ProjMPSContainer([
    TCIA.ProjMPS(Float64, patch, sites) for patch in chi0_patches.data
])
full_mps = TCIA.ProjMPSContainer([
    TCIA.ProjMPS(Float64, patch, sites) for patch in full_patches.data
])
gamma_mps = TCIA.ProjMPSContainer([
    TCIA.ProjMPS(Float64, patch, sites) for patch in gamma_patches.data
])

##

sitesxy_vec = [[x, y] for (x, y) in zip(sitesx, sitesy)]
sitesz_vec = [[z] for z in sitesz]
sites_separatez = [x for pair in zip(sitesxy_vec, sitesz_vec) for x in pair]

function rearrange_siteinds(projmps::TCIA.ProjMPS, sites)
    mps_rearranged = rearrange_siteinds(projmps.data, sites)
    proj = projmps.projector
    sitedims_new = [dim.(s) for s in sites]

    flat_proj = reduce(vcat, proj)
    lengths_new = map(length, sites)

    proj_new_raw = Vector{Vector{Int}}(undef, length(lengths_new))
    start_idx = 1
    for (i, len) in enumerate(lengths_new)
        end_idx = start_idx + len - 1
        proj_new_raw[i] = flat_proj[start_idx:end_idx]
        start_idx = end_idx + 1
    end
    proj_new = TCIA.Projector(proj_new_raw, sitedims_new)
    return TCIA.ProjMPS(mps_rearranged, sites, proj_new)
end

function rearrange_siteinds(projmpss::TCIA.ProjMPSContainer, sites)
    return TCIA.ProjMPSContainer([
        rearrange_siteinds(projmps, sites) for projmps in projmpss.data
    ])
end

full_xy_z = rearrange_siteinds(full_mps, sites_separatez)

full_xy_z.data

##

# multiplication Φ = Γ X₀ F
chi0_full = TCIA.adaptivematmul(chi0_patches, full_patches, pordering; maxbonddim)
phi_bse = TCIA.adaptivematmul(gamma_patches, chi0_full, pordering; maxbonddim)

# normal multiplication for comparison
box = [(x, y) for x in (-halfN):(halfN - 1), y in (-halfN):(halfN - 1)]
chi0_exact = map(splat(fq_chi0), box)
full_exact = map(splat(fq_full), box)
gamma_exact = map(splat(fq_gamma), box)
phi_normalmul = gamma_exact * chi0_exact * full_exact

phi_adaptivemul = [phi_bse(QG.origcoord_to_quantics(grid, p)) for p in box]

@test isapprox(phi_normalmul, phi_adaptivemul; rtol=1e-5)
# end
