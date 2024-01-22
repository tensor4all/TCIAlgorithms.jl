using TCIAlgorithms
using Test

#include("tensor_tests.jl")
#include("elementwisemul_tests.jl")
#include("matrixmul_tests.jl")
#include("matrixmulsum_tests.jl")
#include("matrixmulsum_tests.jl")
#include("partitioned_tensortrain_tests.jl")
include("divideandconquer/adaptivepartitioning_tests.jl")

#using ReTestItems: runtests, @testitem
#using TCIAlgorithms: TCIAlgorithms
##
#runtests(TCIAlgorithms)