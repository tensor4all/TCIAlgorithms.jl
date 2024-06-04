
# Define a mutable struct for TreeNode with a type parameter V for the value
mutable struct TreeNode{V}
    path::Vector{Int}
    value::Union{V, Nothing}
    children::Dict{Vector{Int}, TreeNode{V}}
end

# Function to create a new node
function create_node(::Type{V}, path::Vector{Int}, value::Union{V, Nothing}) where V
    TreeNode{V}(deepcopy(path), value, Dict{Vector{Int}, TreeNode{V}}())
end

function create_node(path::Vector{Int}, value::V) where V
    TreeNode{V}(deepcopy(path), value, Dict{Vector{Int}, TreeNode{V}}())
end

function create_node(::Type{V}, path::Vector{Int}) where V
    TreeNode{V}(deepcopy(path), nothing, Dict{Vector{Int}, TreeNode{V}}())
end

# Function to add a node to the tree
function add_node!(root::TreeNode{V}, path::Vector{Int}, value::Union{V, Nothing}) where V
    path = deepcopy(path)
    current = root
    for i in 1:length(path)
        # Get the current path
        current_path = path[1:i]
        # If the node does not exist, create a new node with nothing value
        if !(current_path in keys(current.children))
            current.children[current_path] = create_node(V, current_path, nothing)
        end
        # Move to the next node
        current = current.children[current_path]
    end
    # Set the value at the last node
    current.value = value
end

# Function to print the tree
function print_tree(node::TreeNode{V}, indent::Int = 0) where V
    value_str = node.value === nothing ? "nothing" : string(node.value)
    println(repeat(" ", indent * 2) * "Path: " * string(node.path) * ", Value: " * value_str)
    for child in values(node.children)
        print_tree(child, indent + 1)
    end
end

# Function to find a node by its path
function find_node(root::TreeNode{V}, path::Vector{Int}) where V
    current = root
    for i in 1:length(path)
        current_path = path[1:i]
        if current_path in keys(current.children)
            current = current.children[current_path]
        else
            return nothing
        end
    end
    return current
end
