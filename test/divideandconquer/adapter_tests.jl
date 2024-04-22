using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

@testset "TTAdapter" begin
    L = 3
    T = Float64
    linkdims = [1, 2, 2, 1]
    sitedims = [[2, 2], [2, 2], [2, 2]]
    #localdims = collect(prod.(sitedims))

    tt = TCI.TensorTrain{T,4}(
        [randn(T, linkdims[l], sitedims[l]..., linkdims[l+1]) for l in 1:L]
    )

    gproj = TCIA.Projector(fill([0, 0], L), sitedims)
    tteval = TCIA.TTAdapter(TCIA.ProjectedTensorTrain(tt, gproj))

    @test tteval([1, 1, 1]) ≈ tt([[1, 1], [1, 1], [1, 1]])
    @test tteval([1, 1, 1]) ≈ tt([[1, 1], [1, 1], [1, 1]])
    @test tteval([2, 2, 2]) ≈ tt([[2, 1], [2, 1], [2, 1]])
    @test tteval([3, 3, 3]) ≈ tt([[1, 2], [1, 2], [1, 2]])

    #@show tteval([[1]], [[1]], Val(1))

end