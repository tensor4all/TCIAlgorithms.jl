using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using Quantics

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul, makeprojectable

using ITensors

@testset "itensor_mul" begin
    @testset "contract (xk-y-z)" begin
        R = 3
        sitesx = [Index(2, "Qubit,x=$n") for n = 1:R]
        sitesk = [Index(2, "Qubit,k=$n") for n = 1:R]
        sitesy = [Index(2, "Qubit,y=$n") for n = 1:R]
        sitesz = [Index(2, "Qubit,z=$n") for n = 1:R]
    
        sitesa = collect(collect.(zip(sitesx, sitesk, sitesy)))
        sitesb = collect(collect.(zip(sitesy, sitesz)))

        p1 = TCIA.ProjMPS(_random_mpo(sitesa), sites, Dict(sitesx[1] => 1))
        p2 = TCIA.ProjMPS(_random_mpo(sitesb), sites, Dict(sitesz[1] => 1))


        #ab_ref = contract(a, b; alg = "naive")
        #ab = FMPOC.contract_densitymatrix(a, b)
        #@test ab_ref ≈ ab
    end
end