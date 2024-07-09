using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA

import TCIAlgorithms: Projector, project, ProjTensorTrain, LazyMatrixMul, makeprojectable

@testset "mul" begin
    @testset "lazymatmul" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1

        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]

        a = TCI.TensorTrain([
            rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
        ])
        b = TCI.TensorTrain([
            rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
        ])

        projector_a = Projector([[1, 1], [0, 0], [0, 0], [0, 0]], sitedimsa)
        #pa = project(ProjTensorTrain(a), projector_a)
        pa = project(makeprojectable(a), projector_a)

        projector_b = Projector([[1, 2], [0, 0], [0, 0], [0, 0]], sitedimsb)
        pb = project(makeprojectable(b), projector_b)

        ab = TCIA.lazymatmul(pa, pb)

        @test ab.sitedims == [[2, 3] for _ in 1:N]

        a_tt = TCI.TensorTrain{T,4}(ab.a.data, sitedimsa)
        b_tt = TCI.TensorTrain{T,4}(ab.b.data, sitedimsb)

        ab_ref = TCI.contract_naive(a_tt, b_tt)

        @test TCIA.fulltensor(ab) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))

        # Fit algorithm
        @test TCIA.fulltensor(TCIA.approxtt(ab)) ≈ TCIA.fulltensor(ProjTensorTrain(ab_ref))
    end

    @testset "projecting lazymul" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1

        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]

        a = TCI.TensorTrain([
            rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
        ])
        b = TCI.TensorTrain([
            rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
        ])

        for p in [[[1, 1], [0, 0], [0, 0], [0, 0]], [[1, 0], [2, 0], [0, 0], [0, 0]]]
            ab = TCIA.lazymatmul(makeprojectable(a), makeprojectable(b))
            _test_projection(ab, Projector(p, sitedimsab))
        end
    end

    @testset "batchevaluate" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1
    
        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]
    
        a = TCI.TensorTrain([
                rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
            ])
        b = TCI.TensorTrain([
                rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
            ])
    
        projector_a = Projector([[1, 1], [1, 0], [0, 0], [0, 0]], sitedimsa)
        pa = project(makeprojectable(a), projector_a)
    
        projector_b = Projector([[1, 2], [0, 1], [0, 0], [0, 0]], sitedimsb)
        pb = project(makeprojectable(b), projector_b)
    
        ab = TCIA.lazymatmul(pa, pb)
    
        leftmmultiidxset = [[[1, 2]]]
        rightmmultiidxset = [[[1, 1]]]
    
        batchmul = TCIA.batchevaluateprj(ab, leftmmultiidxset, rightmmultiidxset, Val(2))
    
        for x in 1:2, y in 1:3 
            @test batchmul[1,1,TCIA._lineari(ab.sitedims[3],[x,y]),1] ≈ ab([[1,2], [1,1], [x,y], [1,1]]) 
        end
    end

    @testset "containermul" begin
        T = Float64
        N = 4
        bonddims = [1, 3, 3, 3, 1]
        @assert length(bonddims) == N + 1
    
        sitedimsa = [[2, 2] for _ in 1:N]
        sitedimsb = [[2, 3] for _ in 1:N]
        sitedimsab = [[2, 3] for _ in 1:N]
    
        pa = ProjTensorTrain{T}[]
        pb = ProjTensorTrain{T}[]
        ab_ref = ProjTensorTrain{T}[]
    
        for i in 1:4
    
            a = TCI.TensorTrain([
                rand(bonddims[n], sitedimsa[n]..., bonddims[n + 1]) for n in 1:N
            ])
            b = TCI.TensorTrain([
                rand(bonddims[n], sitedimsb[n]..., bonddims[n + 1]) for n in 1:N
            ])
    
            projector_a = Projector([TCIA._multii(sitedimsa[1], i), [0, 0], [0, 0], [0, 0]], sitedimsa)
            push!(pa, project(makeprojectable(a), projector_a))

            projector_b = Projector([reverse(TCIA._multii(sitedimsb[1], i)), [0, 0], [0, 0], [0, 0]], sitedimsb)
            push!(pb, project(makeprojectable(b), projector_b))
    

            a_tt = TCI.TensorTrain{T,4}(pa[i].data, sitedimsa)
            b_tt = TCI.TensorTrain{T,4}(pb[i].data, sitedimsb)
            push!(ab_ref, ProjTensorTrain(TCI.contract_naive(a_tt, b_tt)))
    
        end
    
        cont_pa = TCIA.ProjTTContainer(pa)
        cont_pb = TCIA.ProjTTContainer(pb)
    
        cont_ab_ref = TCIA.ProjTTContainer(ab_ref)
    
        cont_ab = TCIA.lazymatmul(cont_pa, cont_pb)
    
       TCIA.fulltensor(cont_ab) ≈ TCIA.fulltensor(cont_ab_ref)
    end

end

