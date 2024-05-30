using Distributed
using Test

const MAX_WORKERS = 4

# Add worker processes if necessary.
addprocs(max(0, MAX_WORKERS - nworkers()))

@everywhere begin
    import TensorCrossInterpolation as TCI
    import TCIAlgorithms as TCIA

    function workerfunc(n)
        if n > 0
            return 1.0 * n, -1 * n
        else
            return 1.0 * n, nothing
        end
    end
end

@testset "TaskQueue" begin
    queue = TCIA.TaskQueue{Int,Float64}(collect(1:10))
    results = TCIA.loop(queue, workerfunc; verbosity=0)

    @test results == union(Set([1.0 * n for n in 1:10]), Set([-1.0 * n for n in 1:10]))
end
