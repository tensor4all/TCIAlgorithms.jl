using Test
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
using Distributed

@everywhere function f(n)
    sleep(0.1)
    if n > 0
        return 1.0 * n, -1 * n
    else
        return 1.0 * n, nothing
    end
end

@testset "TaskQueue" begin
    queue = TCIA.TaskQueue{Int,Float64}(Set(1:10))
    results = TCIA.loop(queue, f)

    @test results == union(Set([1.0 * n for n in 1:10]), Set([-1.0 * n for n in 1:10]))
end