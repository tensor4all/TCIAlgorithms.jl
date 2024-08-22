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

    allpatches = reduce(append!, [node.value for node in all_nodes(root)])

    root_tt = create_node(ProjTensorTrain{T}, Int[])
    for x in allpatches
        add!(root_tt, x, pordering)
    end

    return ProjTTContainer{T}(_mergesmallpatches(root_tt; tolerance, maxbonddim))
end


#==
function matmul(
    a::ProjTTContainer{T},
    b::ProjTTContainer{T},
    bs::BlockStructure;
    tolerance=1e-14,
    maxbonddim=typemax(Int),
)::ProjContainer{T} where {T}
    result = [Set{ProjTensorTrain{T}}() for _ in 1:length(bs.blocks)]
    for x in a, y in b # FIXME: Naive loop over O(N^2) pairs
        xy = lazymatmul(x, y)
        if xy === nothing
            continue
        end
        for (ib, b) in enumerate(bs)
            if hasoverlap(xy.projector, b)
                push!(
                    result[ib],
                    approxtt(project(xy, xy.projector & b); maxbonddim=maxbonddim, tolerance=tolerance)
                )
            end
        end
    end

    ptts = ProjTensorTrain{T}[]
    for (ib, b) in enumerate(bs)
        patch = reduce((x, y) -> add(x, y; tolerance, maxbonddim), result[ib])
        push!(ptts, project(patch, b))
    end

    return ProjTTContainer{T}(ptts)
end
==#


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
end

function _mergesmallpatches(
    node::TreeNode{ProjTensorTrain{T}}; tolerance=1e-14, maxbonddim=typemax(Int)
)::Vector{ProjTensorTrain{T}} where {T}
    # The following implementation is based on
    # recursive merging of patches, which is not efficient for paralellization.
    tt_child::Vector{ProjTensorTrain{T}} = reduce(
        append!,
        (_mergesmallpatches(c; tolerance, maxbonddim) for c in values(node.children));
        init=ProjTensorTrain{T}[],
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

    sum_value = reduce((x, y) -> add(x, y; tolerance, maxbonddim), all_values)

    if maximum(TCI.linkdims(sum_value.data)) == maxbonddim
        # Unsafe to merge
        return all_values
    else
        # Safe to merge
        return [sum_value]
    end
end

"""
Perform matrix multiplication of two tensor trains and project the result to a block structure.
"""
function matmul(
    a::ProjTTContainer{T},
    b::ProjTTContainer{T},
    bs::BlockStructure;
    tolerance=1e-14,
    maxbonddim=typemax(Int),
) where {T}
    tts = [zeroprojtt(T, prj) for prj in bs]
    for x in a, y in b # FIXME: Naive loop over O(N^2) pairs
        xy = lazymatmul(x, y)
        if xy === nothing
            continue
        end
        for (ib, p) in enumerate(bs.blocks)
            if hasoverlap(xy.projector, p)
                tt_ = approxtt(project(xy, p); maxbonddim=maxbonddim, tolerance=tolerance)
                tts[ib] = add(tts[ib], tt_; tolerance, maxbonddim)
            end
        end
    end

    return ProjTTContainer(tts)
end
