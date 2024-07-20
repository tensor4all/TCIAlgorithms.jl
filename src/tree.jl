
# Define a mutable struct for TreeNode with a type parameter V for the value
mutable struct TreeNode{V}
    path::Vector{Int}
    value::Vector{V}
    children::Dict{Vector{Int},TreeNode{V}}
end

_isvalidpath(path::Vector{Int}) = all(path .> 0)

# Function to create a new node
function create_node(path::Vector{Int}, value::V) where {V}
    _isvalidpath(path) || throw(ArgumentError("Invalid path $path"))
    return TreeNode{V}(deepcopy(path), [value], Dict{Vector{Int},TreeNode{V}}())
end

function create_node(::Type{V}, path::Vector{Int}) where {V}
    _isvalidpath(path) || throw(ArgumentError("Invalid path $path"))
    return TreeNode{V}(deepcopy(path), V[], Dict{Vector{Int},TreeNode{V}}())
end

# Function to add a value to the tree
function add_value!(node::TreeNode{V}, path::Vector{Int}, value::V) where {V}
    _isvalidpath(path) || throw(ArgumentError("Invalid path $path"))

    path[1:length(node.path)] == node.path ||
        error("path $path does not match node path $(node.path)")

    path = deepcopy(path)
    current = node
    for i in (length(node.path) + 1):length(path)
        # Get the current path
        current_path = path[1:i]
        # If the node does not exist, create a new node with nothing value
        if !(current_path in keys(current.children))
            current.children[current_path] = create_node(V, current_path)
        end
        # Move to the next node
        current = current.children[current_path]
    end
    # Set the value at the last node
    return push!(current.value, value)
end

"""
Remove a value at a given path
"""
function delete_value!(root::TreeNode{V}, path::Vector{Int}, value::V)::Vector{V} where {V}
    node = find_node(root, path)
    if node === nothing
        error("Not found $path")
    end
    matches = findall(x -> x == value, node.value)
    if length(matches) == 0
        error("Not found $value at $path")
    elseif length(matches) > 1
        error("Multiple matches found")
    else
        return deleteat!(node.value, matches[1])
    end
end

"""
Remove a value at a given path
"""
function delete_node!(root::TreeNode{V}, path::Vector{Int}) where {V}
    parentnode = find_node(root, path[1:(end - 1)])
    if parentnode === nothing
        error("Not found parent node for $path")
    end
    return Base.delete!(parentnode.children, path)
end

# Function to print the tree
function print_tree(io::IO, node::TreeNode{V}, indent::Int=0) where {V}
    value_str = length(node.value) == 0 ? "nothing" : string(node.value)
    println(
        io, repeat(" ", indent * 2) * "Path: " * string(node.path) * ", Value: " * value_str
    )
    for child in values(node.children)
        print_tree(io, child, indent + 1)
    end
end

# Function to find a node by its path
function find_node(node::TreeNode{V}, path::Vector{Int}) where {V}
    node.path == path[1:length(node.path)] ||
        error("path $path does not match node path $(node.path)")
    current = node
    for i in (length(node.path) + 1):length(path)
        current_path = path[1:i]
        if current_path in keys(current.children)
            current = current.children[current_path]
        else
            return nothing
        end
    end
    return current
end

# Iterator to visit all nodes
struct TreeNodeIterator{V}
    stack::Vector{TreeNode{V}}
end

function Base.iterate(iter::TreeNodeIterator{V}) where {V}
    isempty(iter.stack) && return nothing
    current = pop!(iter.stack)
    append!(iter.stack, values(current.children))
    return current, iter
end

function Base.iterate(iter::TreeNodeIterator{V}, state) where {V}
    return iterate(state)
end

# Implementing length for TreeNodeIterator
function Base.length(iter::TreeNodeIterator{V}) where {V}
    count = 0
    for _ in all_nodes(iter.stack[1])
        count += 1
    end
    return count
end

function all_nodes(node::TreeNode{V}) where {V}
    return TreeNodeIterator([node])
end

function isleaf(node::TreeNode{V}) where {V}
    return length(node.children) == 0
end
