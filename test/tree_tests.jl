using Test
using Random

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
import TCIAlgorithms as TCIA
import TCIAlgorithms: create_node, add_node!, find_node, all_nodes, delete!

@testset "tree" begin
    @testset "add_node! and find_node" begin
        root = create_node(Int[], "root")

        add_node!(root, Int[1], "child1")
        add_node!(root, [1, 2, 1], "grandchild1")
        add_node!(root, Int[2, 3], "child3")

        # Check root node
        @test root.path == Int[]
        @test root.value == ["root"]

        # Check child nodes
        node = find_node(root, [1])
        @test node !== nothing
        @test node.value == ["child1"]

        node = find_node(root, [1, 2, 1])
        @test node !== nothing
        @test node.value == ["grandchild1"]
    end

    @testset "add_node! (from root)" begin
        root = create_node([1], "root")
        add_node!(root, [1, 2, 1, 1], "child1")
        @test find_node(root, [1, 2, 1, 1]).value == ["child1"]
        delete!(root, [1, 2, 1, 1], "child1")
        @test find_node(root, [1, 2, 1, 1]).value == String[]
    end

    @testset "add_node! (from non-root)" begin
        root = create_node([1], "root")
        add_node!(root, [1, 2, 3], "child1")
        node = find_node(root, [1, 2])
        @test find_node(node, [1, 2, 3]).value == ["child1"]
        @test delete!(root, [1, 2, 3], "child1") == String[]
    end

    @testset "all_nodes" begin
        # Example usage of the tree and iterator
        root = create_node([1], "root")

        add_node!(root, [1, 2, 1, 1], "child1")
        add_node!(root, [1, 2, 2, 1], "child2")
        add_node!(root, [1, 2, 2], "child3")
        add_node!(root, [1, 3, 1, 1], "child4")
        add_node!(root, [1, 3, 2, 1], "child5")

        @test Set(
            reduce(
                append!, node.value for node in all_nodes(root) if length(node.value) != 0
            ),
        ) == Set(["root", "child1", "child2", "child3", "child4", "child5"])
    end

    @testset "delete!" begin
        root = create_node(Int[], "root")

        add_node!(root, [1], "child1")
        add_node!(root, [1, 2, 1], "grandchild1")
        add_node!(root, [2, 3], "child3")

        delete!(root, [1, 2, 1], "grandchild1")

        @test Set(["child1", "root", "child3"]) ==
            Set(reduce(append!, (node.value for node in all_nodes(root))))
    end
end