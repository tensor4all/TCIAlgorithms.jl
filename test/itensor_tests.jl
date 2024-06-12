using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul, makeprojectable

using ITensors

@testset "itensor" begin
    @testset "ProjMPS" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        Ψ = MPS(collect(_random_mpo(sites)))
        prjΨ = TCIA.ProjMPS(Ψ, sites)

        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))
        prjΨ2 = project(prjΨ, Dict(sitesx[1] => 2))

        Ψreconst = MPS(prjΨ1) + MPS(prjΨ2)

        @test Ψreconst ≈ Ψ
    end

    @testset "conversion" begin
        N = 2
        sitesx = [Index(2, "x=$n") for n in 1:N]
        sitesy = [Index(2, "y=$n") for n in 1:N]
        sites = collect(collect.(zip(sitesx, sitesy)))
        sitedims = [dim.(s) for s in sites]
        Ψ = MPS(collect(_random_mpo(sites)))
        prjΨ = TCIA.ProjMPS(Ψ, sites)
        prjΨ1 = project(prjΨ, Dict(sitesx[1] => 1))

        prjtt1 = TCIA.ProjTensorTrain{Float64}(prjΨ1)
        @test prjtt1.projector == Projector([[1, 0], [0, 0]], sitedims)

        prjΨ1_reconst = TCIA.ProjMPS(Float64, prjtt1, sites)

        @test prjΨ1 ≈ prjΨ1_reconst
    end
end
