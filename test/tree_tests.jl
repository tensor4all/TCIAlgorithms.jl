using Test
using Random

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
import TCIAlgorithms: create_node, add_node!, find_node

@testset "TreeNode Tests" begin
    root = create_node([0, 0, 0], "root")

    add_node!(root, [1, 0, 0], "child1")
    add_node!(root, [1, 2, 1], "grandchild1")
    add_node!(root, [2, 3, 0], "child3")

    # Check root node
    @test root.path == [0, 0, 0]
    @test root.value == ["root"]

    # Check child nodes
    node = find_node(root, [1, 0, 0])
    @test node !== nothing
    @test node.value == ["child1"]

    #node = find_node(root, [1, 2, 0])
    #@test node !== nothing
    #@test node.value === nothing

    node = find_node(root, [1, 2, 1])
    @test node !== nothing
    @test node.value == ["grandchild1"]

    #node = find_node(root, [2, 0, 0])
    #@test node !== nothing
    #@test node.value === nothing
end