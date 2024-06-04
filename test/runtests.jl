using Distributed

using TCIAlgorithms
import TCIAlgorithms as TCIA
using Test

@everywhere gaussian(x, y) = exp(-0.5 * (x^2 + y^2))
const MAX_WORKERS = 2

# Add worker processes if necessary.
if nworkers() < MAX_WORKERS
    addprocs(max(0, MAX_WORKERS - nworkers()))
end

include("_util.jl")

#include("util_tests.jl")
#include("projector_tests.jl")
#include("projectable_evaluator_tests.jl")
#include("projtensortrain_tests.jl")
#include("container_tests.jl")
#include("mul_tests.jl")
#include("distribute_tests.jl")
include("patching_tests.jl")
#include("crossinterpolate_tests.jl")
#include("crossinterpolate_lazyeval_tests.jl")
include("tree_tests.jl")
