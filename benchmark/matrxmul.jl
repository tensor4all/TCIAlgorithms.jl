import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using BenchmarkTools
using Random

function print_benchmark_result(benchmark_result)
    println("Minimum time: ", minimum(benchmark_result).time, " ns")
    println("Median time: ", median(benchmark_result).time, " ns")
    println("Mean time: ", mean(benchmark_result).time, " ns")
    println("Maximum time: ", maximum(benchmark_result).time, " ns")
    println("Allocations: ", benchmark_result.allocs)
    return println("Memory usage: ", benchmark_result.memory, " bytes")
end

function _random_indexset(localdims, nsample)
    result = Set{Vector{Int}}()
    if nsample > 0.1 * prod(localdims)
        error("Error: nsample is too large")
    end
    while length(result) < nsample
        candidate = [rand(1:d) for d in localdims]
        if candidate ∉ result
            push!(result, candidate)
        end
    end
    res_ = collect(result)
    @assert length(res_) == nsample
    return res_
end

function benchmark1()
    N = 40
    χ = 100
    bonddims_a = vcat(1, fill(χ, N - 1), 1)
    bonddims_b = vcat(1, fill(χ, N - 1), 1)
    localdims1 = fill(2, N)
    localdims2 = fill(2, N)
    localdims3 = fill(2, N)

    a = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_a[n], localdims1[n], localdims2[n], bonddims_a[n + 1]) for
        n in 1:N
    ])
    b = TCI.TensorTrain{ComplexF64,4}([
        rand(ComplexF64, bonddims_b[n], localdims2[n], localdims3[n], bonddims_b[n + 1]) for
        n in 1:N
    ])

    ab = TCIA.LazyMatrixMul(a, b)

    nl = N ÷ 2
    nr = N - nl - 2
    ncent = N - nl - nr
    @show nl, nr, ncent
    @assert ncent > 0

    χ2 = χ
    leftindexset = _random_indexset(localdims1[1:nl], χ2)
    rightindexset = _random_indexset(localdims1[(N - nr + 1):N], χ2)

    # Compile & cache
    ab(leftindexset, rightindexset, Val(ncent))

    # Measurement
    t1 = time_ns()
    ab(leftindexset, rightindexset, Val(ncent))
    t2 = time_ns()
    return println("Runtime time: ", (t2 - t1) * 1e-9, " sec")
end

benchmark1()
