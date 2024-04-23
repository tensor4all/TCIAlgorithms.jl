using TCIAlgorithms
using Test

include("util_tests.jl")
include("tensor_tests.jl")
include("elementwisemul_tests.jl")
include("matrixmul_tests.jl")
include("matrixmulsum_tests.jl")

include("divideandconquer/projector_tests.jl")
include("divideandconquer/partitioned_tensortrain_tests.jl")
include("divideandconquer/adaptivepartitioning_tests.jl")

#include("divideandconquer/adapter_tests.jl")
