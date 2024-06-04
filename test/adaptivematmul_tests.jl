using Test
using Random

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

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
    adaptivematmul

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
end