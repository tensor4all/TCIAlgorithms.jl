#using TCIAlgorithms
#using Test

#include("test_tensor.jl")
#include("test_elementwisemul.jl")
#include("test_matrixmul.jl")
#include("test_matrixmulsum.jl")


using ReTestItems: runtests, @testitem
using TCIAlgorithms: TCIAlgorithms

runtests(TCIAlgorithms)