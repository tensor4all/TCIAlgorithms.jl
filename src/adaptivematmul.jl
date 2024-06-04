function adaptivematmul(
    a::ProjTTContainer{T}, b::ProjTTContainer{T}, pordering::PatchOrdering;
    tolerance=1e-14, maxbonddim=typemax(Int)
) where {T}
    NT = Union{ProjTensorTrain{T},LazyMatrixMul{T}}

    root = create_node(NT, Int[])
    for x in a, y in b # FIXME: Naive loop over O(N^2) pairs
        xy = lazymatmul(x, y)
        if xy !== nothing
            add_node!(root, xy, pordering)
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
    pordering::PatchOrdering
    ; 
    tolerance=1e-14, maxbonddim=typemax(Int)
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
                if maximum(TCI.linkdims(vresult.data)) < maxbonddim || length(node.path) == L
                    # Bond dimension is small enough or no way to project more!
                    @assert delete!(node, node.path, v) !== nothing
                    add_node!(node, node.path, vresult)
                else
                    @assert delete!(node, node.path, v) !== nothing
                    for proj in makechildproj(v.projector, pordering)
                        #println("")
                        v_ = project(v, proj)
                        add_node!(node, createpath(proj, pordering), v_)
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
    root
    #return ProjTTContainer{T}(reduce(append!, [node.value for node in all_nodes(root)]))
end