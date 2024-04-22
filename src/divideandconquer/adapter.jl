"""
Evaluating a TT with an arbitrary number of site indices as a TT with one site index.
This is useful for a TCI of a TT with more than one site index.
"""
struct TTAdapter{T} <: TCI.BatchEvaluator{T}
    tt::ProjectableEvaluator{T}
    sitedims::Vector{Vector{Int}}
end

function TTAdapter(tt::ProjectableEvaluator{T}) where {T}
    return TTAdapter{T}(tt, tt.sitedims)
end

function (obj::TTAdapter{T})(
    indexset::MultiIndex
)::T where {T}
    return obj.tt(multii(obj.sitedims, indexset))
end

function (obj::TTAdapter{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end
    NL  = length(leftindexset[1])
    NR  = length(rightindexset[1])
    leftindexset_ = [multii(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [multii(obj.sitedims[end-NR+1:end], x) for x in rightindexset]

    return obj.tt(leftindexset_, rightindexset_, Val(M))
end