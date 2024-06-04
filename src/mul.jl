"""
Lazy evaluation for matrix multiplication of two TTOs

This is for the lazy evaluation of the matrix multiplication of two TTOs.
If underlying data needs to be changed by projection, it will be copied.

TODO: make immutable
"""
mutable struct LazyMatrixMul{T} <: ProjectableEvaluator{T}
    coeff::T
    contraction::TCI.Contraction{T}
    projector::Projector
    sitedims::Vector{Vector{Int}}
    a::ProjTensorTrain{T}
    b::ProjTensorTrain{T}
end

function LazyMatrixMul{T}(
    a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(T)
) where {T}
    # This restriction is due to simulicity and to be removed.
    all(length.(a.sitedims) .== 2) || error("The number of site indices must be 2")
    all(length.(b.sitedims) .== 2) || error("The number of site indices must be 2")
    sitedims_ab = [[x[1], y[2]] for (x, y) in zip(a.sitedims, b.sitedims)]
    a_tt = TensorTrain{T,4}(a.data, a.sitedims)
    b_tt = TensorTrain{T,4}(b.data, b.sitedims)
    contraction = TCI.Contraction(a_tt, b_tt)
    projector = Projector(
        [[x[1], y[2]] for (x, y) in zip(a.projector, b.projector)], sitedims_ab
    )
    return LazyMatrixMul{T}(coeff, contraction, projector, sitedims_ab, a, b)
end

function LazyMatrixMul(a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(1)) where {T}
    return LazyMatrixMul{T}(a, b; coeff=coeff)
end

Base.length(obj::LazyMatrixMul) = length(obj.projector)

function project(
    obj::LazyMatrixMul{T}, prj::Projector; kwargs...
)::LazyMatrixMul{T} where {T}
    projector_a_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(prj, obj.a.projector)], obj.a.sitedims
    )
    projector_b_new = Projector(
        [[x[1], y[2]] for (x, y) in zip(obj.b.projector, prj)], obj.b.sitedims
    )
    obj.a = project(obj.a, projector_a_new; kwargs...)
    obj.b = project(obj.b, projector_b_new; kwargs...)
    # TO BE FIXED: Cache is thrown away
    return LazyMatrixMul{T}(obj.a, obj.b; coeff=obj.coeff)
end

function lazymatmul(
    a::ProjTensorTrain{T}, b::ProjTensorTrain{T}; coeff=one(T)
)::Union{LazyMatrixMul{T},Nothing} where {T}
    if !hasoverlap((x[2] for x in a.projector), (x[1] for x in b.projector))
        return nothing
    end
    return LazyMatrixMul{T}(a, b; coeff=coeff)
end

function LazyMatrixMul(a::ProjTensorTrain, b::ProjTensorTrain)
    return LazyMatrixMul(TCI.Contraction(a, b))
end

# multi-site-index evaluation
function (obj::LazyMatrixMul{T})(indexset::MMultiIndex)::T where {T}
    return obj.contraction(lineari(obj.sitedims, indexset))
end

# multi-site-index evaluation
function batchevaluateprj(
    obj::LazyMatrixMul{T},
    leftindexset::AbstractVector{MMultiIndex},
    rightmmultiidxset::AbstractVector{MMultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    #if !all([all(p .== 0) for p in obj.projector.data])
        #@show obj.projector.data
    #end
    M >= 0 || error("The order of the result must be non-negative")
    if length(leftindexset) * length(rightmmultiidxset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    NL = length(leftindexset[1])
    NR = length(rightmmultiidxset[1])
    L = length(obj)
    leftindexset_ = [lineari(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [
        lineari(obj.sitedims[(end - NR + 1):end], x) for x in rightmmultiidxset
    ]
    projector = Vector{Int}[
        isprojectedat(obj.projector, n) ? obj.projector[n] : [0, 0]
        for n in (NL + 1):(L - NR)
    ]
    returnshape = [
        isprojectedat(obj.projector, n) ? 1 : prod(obj.sitedims[n]) for
        n in (NL + 1):(L - NR)
    ]
    res = TCI.batchevaluate(
        obj.contraction, leftindexset_, rightindexset_, Val(M), projector
    )
    return reshape(res, length(leftindexset), returnshape..., length(rightmmultiidxset))
end

function lazymatmul(
    a::ProjTTContainer{T}, b::ProjTTContainer{T}
)::ProjContainer{T,LazyMatrixMul{T}} where {T}
    muls = LazyMatrixMul{T}[]
    for l in a, r in b
        p = lazymatmul(l, r)
        if p !== nothing
            push!(muls, p)
        end
    end
    return ProjContainer{T,LazyMatrixMul{T}}(muls)
end

function approxtt(
    obj::LazyMatrixMul{T}; maxbonddim=typemax(Int), tolerance=1e-14, kwargs...
)::Union{ProjTensorTrain{T},Nothing} where {T}
    sites_a_dangling = [Index(ld[1], "ell=$ell, a_dangling") for (ell, ld) in enumerate(obj.a.sitedims)]
    sites_ab_shared = [Index(ld[2], "ell=$ell, ab_shared") for (ell, ld) in enumerate(obj.a.sitedims)]
    sites_b_dangling = [Index(ld[2], "ell=$ell, b_dangling") for (ell, ld) in enumerate(obj.b.sitedims)]
    sites_a = collect(collect(x) for x in zip(sites_a_dangling, sites_ab_shared))
    sites_b = collect(collect(x) for x in zip(sites_ab_shared, sites_b_dangling))
    sites_ab = collect(collect(x) for x in zip(sites_a_dangling, sites_b_dangling))

    a_tto = TensorTrain{T,4}(obj.a.data, obj.a.sitedims)
    b_tto = TensorTrain{T,4}(obj.b.data, obj.b.sitedims)

    a_MPO = MPO(a_tto; sites=sites_a)
    b_MPO = MPO(b_tto; sites=sites_b)

    ab_MPO = obj.coeff * FMPOC.contract_fit(a_MPO, b_MPO; maxdim=maxbonddim, cutoff=tolerance^2)
    ab_tto = TensorTrain{T,4}(ab_MPO; sites=sites_ab)

    return project(ProjTensorTrain(ab_tto), obj.projector)
end

function isapproxttavailable(obj::LazyMatrixMul)
    return true
end
