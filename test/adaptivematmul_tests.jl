using Test
using Random

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: create_node, add!, find_node, all_nodes, delete!, ProjTensorTrain, Projector, project, ProjTTContainer, adaptivematmul

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
        abmat ≈ amat * bmat
    end
end