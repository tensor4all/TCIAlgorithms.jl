# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(indexset::MultiIndex)::T where {T}
    return obj(multii(obj.sitedims, indexset))
end

# single-site-index evaluation
function (obj::ProjectableEvaluator{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{T,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    NL = length(leftindexset[1])
    NR = length(rightindexset[1])
    leftindexset_ = [multii(obj.sitedims[1:NL], x) for x in leftindexset]
    rightindexset_ = [multii(obj.sitedims[(end - NR + 1):end], x) for x in rightindexset]

    return obj(leftindexset_, rightindexset_, Val(M))
end
