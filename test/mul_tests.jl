using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector

@testset "projectat!" begin
    N = 4
    bonddims = [1, 3, 3, 3, 1]
    sitedims = [[2, 2] for _ in 1:N]
    @assert length(bonddims) == N + 1

    a = TCI.TensorTrain([
        rand(bonddims[n], 2, 2, bonddims[n + 1]) for n in 1:N
    ])
    b = TCI.TensorTrain([
        rand(bonddims[n], 2, 2, bonddims[n + 1]) for n in 1:N
    ])
end