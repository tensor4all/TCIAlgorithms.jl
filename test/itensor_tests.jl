using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul, makeprojectable

using ITensors

@testset "itensor" begin
    @testset "ProjMPO" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        Ψ = _random_mpo(sites)
        prjΨ = TCIA.ProjMPO(Ψ, sites)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = MPO(prjΨ1) + MPO(prjΨ2)

        @test Ψreconst ≈ Ψ
    end
end
