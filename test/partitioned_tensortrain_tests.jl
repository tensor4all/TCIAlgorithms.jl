@testitem "partitioned_tensortrain_tests.jl" begin

    using Test
    import TensorCrossInterpolation as TCI
    import TCIAlgorithms as TCIA

    @testset "project_is_compatible" begin
        @test_throws ErrorException TCIA.iscompatible([0, 0, 0], [0])
        @test TCIA.iscompatible([0, 0, 0], [0, 0, 0]) == true
        @test TCIA.iscompatible([0, 1, 0], [0, 2, 0]) == false
        @test TCIA.iscompatible([0, 1, 0], [0, 0, 1]) == true
    end

    @testset "projectat!" begin
        A_org = ones(2, 2, 2)

        let
            A = copy(A_org)
            TCIA.projectat!(A, 1, 1)
            @test all(A[1, :, :] .== A_org)
            @test all(A[2, :, :] .== 0.0)
        end

        let
            A = copy(A_org)
            TCIA.projectat!(A, 1, 2)
            @test all(A[1, :, :] .== 0.0)
            @test all(A[2, :, :] .== A_org)
        end

        let
            A = copy(A_org)
            TCIA.projectat!(A, 2, 1)
            @test all(A[:, 1, :] .== A_org)
            @test all(A[:, 2, :] .== 0.0)
        end


    end

    @testset "truncate" begin
        using Random
        Random.seed!(100)
        N = 4
        bonddims = [1, 30, 30, 30, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [2, 2, 2, 2]

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n+1]) for n = 1:N
        ])

        tt2 = TCIA.truncate(tt; cutoff=1e-30)

        indices = [[[rand(1:localdims1[n]), rand(1:localdims2[n])] for n = 1:N] for _ in 1:100]
        before = [tt(idx) for idx in indices]
        after = [tt2(idx) for idx in indices]
        @test before ≈ after
    end


    @testset "ProjectedTensorTrain" begin
        N = 4
        bonddims = [1, 10, 10, 10, 1]
        @assert length(bonddims) == N + 1

        localdims1 = [2, 2, 2, 2]
        localdims2 = [3, 3, 3, 3]

        tt = TCI.TensorTrain([
            rand(bonddims[n], localdims1[n], localdims2[n], bonddims[n+1]) for n = 1:N
        ])

        # Projection 
        #prj = [[1, 0, 0, 0], [0, 0, 0, 0]]
        prj = [[1, 0], [0, 0], [0, 0], [0, 0]]
        prj_data = deepcopy(tt.T)
        prj_data[1][:, 2, :, :] .= 0.0
        ptt = TCIA.ProjectedTensorTrain{Float64,4}(tt, prj)

        # Within the partition
        indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
        @test tt(indexset1) == ptt(indexset1) # exact equality

        # Outside the partition
        indexset2 = [[2, 1], [1, 1], [1, 1], [1, 1]]
        @test ptt(indexset2) == 0.0

        # Evaluation at a single linear indexset
        indexset3 = [[1, 1], [1, 1], [1, 1], [2, 1]]
        indexset3_li = [1, 1, 1, 2]
        @test TCIA._multii(ptt, indexset3_li) == indexset3
        @test ptt(indexset3) == ptt(indexset3_li)

        # Projection with truncation
        ptt_truncated = TCIA.ProjectedTensorTrain{Float64,4}(tt)
        TCIA.project!(ptt_truncated, prj; compression=true)
        indexset1 = [[1, 1], [1, 1], [1, 1], [1, 1]]
        @test tt(indexset1) ≈ ptt_truncated(indexset1) # exact equality
    end
end