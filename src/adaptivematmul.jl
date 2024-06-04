function adaptivematmul(
    a::ProjTTContainer{T},
    b::ProjTTContainer{T},
    pordering::PatchOrdering;
    tolerance=1e-14,
    maxbonddim=typemax(Int),
) where {T}
    NT = Union{ProjTensorTrain{T},LazyMatrixMul{T}}

    root = create_node(NT, Int[])
    for x in a, y in b # FIXME: Naive loop over O(N^2) pairs
        xy = lazymatmul(x, y)
        if xy !== nothing
            add!(root, xy, pordering)
        end
    end

    # Perform lazy matrix multiplications
    _performmul!(root, pordering; tolerance=tolerance, maxbonddim=maxbonddim)

    return ProjTTContainer{T}(reduce(append!, [node.value for node in all_nodes(root)]))
end

"""
Perform matrix multiplication of two tensor trains and if the bond dimension is too large, project the result to a lower bond dimension.
We repeat this process until the bond dimension is small enough or no way to project more.
"""
function _performmul!(
    root::TreeNode{Union{ProjTensorTrain{T},LazyMatrixMul{T}}},
    pordering::PatchOrdering;
    tolerance=1e-14,
    maxbonddim=typemax(Int),
) where {T}
    # Check if lazy matrix multiplications can be performed
    L = length(pordering)
    while true
        updated = false
        for node in all_nodes(root)
            for v in node.value
                if !(v isa LazyMatrixMul{T})
                    continue
                end
                updated = true
                vresult = approxtt(v; maxbonddim=maxbonddim, tolerance=tolerance)
                if maximum(TCI.linkdims(vresult.data)) < maxbonddim ||
                    length(node.path) == L
                    # Bond dimension is small enough or no way to project more!
                    @assert delete_value!(node, node.path, v) !== nothing
                    add_value!(node, node.path, vresult)
                else
                    @assert delete_value!(node, node.path, v) !== nothing
                    for proj in makechildproj(v.projector, pordering)
                        #println("")
                        v_ = project(v, proj)
                        add_value!(node, createpath(proj, pordering), v_)
                    end
                end
            end
        end
        if !updated
            break
        end
    end
    for node in all_nodes(root)
        for v in node.value
            v isa ProjTensorTrain{T} || error("Something went wrong!")
        end
    end
    return root
    #return ProjTTContainer{T}(reduce(append!, [node.value for node in all_nodes(root)]))
end

#"""
#Merge TTs until the bond dimensions reaches the upper limit.
#Data will be deepcopied!
#"""
#function mergesmallpatches!(
    #root::TreeNode{ProjTensorTrain{T}}; tolerance=1e-14, maxbonddim=typemax(Int)
#)::ProjTTContainer{T} where {T}
    #done = ProjTensorTrain{T}[]
    #while true
        #num_done = length(done)
        #_mergesmallpatches!(root, done; tolerance, maxbonddim)
        #@show num_done
        #if num_done == length(done)
            #break
        #end
    #end
    #return ProjTTContainer{T}(done)
#end
#
function _mergesmallpatches(
    node::TreeNode{ProjTensorTrain{T}};
    tolerance=1e-14,
    maxbonddim=typemax(Int),
)::Vector{ProjTensorTrain{T}} where {T}
    tt_child::Vector{ProjTensorTrain{T}} = reduce(
        append!,
        (_mergesmallpatches(c; tolerance, maxbonddim) for c in values(node.children));
        init = ProjTensorTrain{T}[]
    )

    all_values = vcat(tt_child, node.value)
    for x in all_values
        if maximum(TCI.linkdims(x.data)) > maxbonddim
            TCI.compress!(x.data; tolerance=0.0, maxbonddim=maxbonddim)
        end
    end

    if length(all_values) == 1
        return all_values
    end

    sum_value = deepcopy(all_values[1])
    for n in 2:length(all_values)
        sum_value = add(sum_value, all_values[n]; tolerance, maxbonddim)
    end

    if maximum(TCI.linkdims(sum_value.data)) == maxbonddim
        return all_values
    else
        return [sum_value]
    end

    #==
    for node in all_nodes(root)
        @show node.path, isleaf(node)
        if isleaf(node) || isempty(node.value)
            continue
        end
        @show node.path, isleaf(node)
        @assert length(node.children) == 0
        @show node.path
        @show length(node.value)
        if length(node.value) == 1
            continue
        end

        sum_value = reduce(
            (x, y) -> add(x, y; tolerance=tolerance, maxbonddim=maxbonddim), node.value
        )

        @assert maximum(TCI.linkdims(sum_value.data)) <= maxbonddim
        @show maximum(TCI.linkdims(sum_value.data)), maxbonddim

        if maximum(TCI.linkdims(sum_value.data)) == maxbonddim
            for v in node.value
                push!(done, v)
                @assert delete_value!(root, node.path, v) !== nothing
            end
            @assert delete_node!(root, node.path) !== nothing
        else
            for v in node.value
                @assert delete_value!(root, node.path, v) !== nothing
            end
            add_value!(root, node.path, sum_value)
        end
    end
    ==#
end