using Test
using Aqua
using JET
using TCIAlgorithms

@testset "Aqua.jl" begin
    Aqua.test_all(TCIAlgorithms; deps_compat=false)
end

#=
@testset "JET.jl" begin
    JET.test_package(TCIAlgorithms; target_defined_modules=true)
end
=#
