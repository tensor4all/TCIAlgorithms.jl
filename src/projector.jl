struct Projector
    data::Vector{Vector{Int}} # 0 means no projection
    sitedims::Vector{Vector{Int}}
    function Projector(data, sitedims)
        for (d, s) in zip(data, sitedims)
            length(d) == length(s) || error("Length mismatch")
            for (d_, s_) in zip(d, s)
                if d_ > s_ || d_ < 0
                    error(
                        "Invalid projector, out of bounds, data: $(data), sitedims: $(sitedims)",
                    )
                end
            end
        end
        return new(data, sitedims)
    end
end

# Reverse the left and right indices
function Base.reverse(obj::Projector)
    return Projector(reverse(obj.data), reverse(obj.sitedims))
end

function Base.copy(obj::Projector)
    return Projector(deepcopy(obj.data), deepcopy(obj.sitedims))
end

function Base.iterate(p::Projector, state=1)
    if state > length(p.data)
        return nothing
    end
    return (p.data[state], state + 1)
end

Base.length(p::Projector) = length(p.data)
Base.getindex(p::Projector, i::Union{Int,AbstractRange{Int},Colon}) = p.data[i]
Base.lastindex(p::Projector) = Base.lastindex(p.data)

function (p::Projector)(isite::Int, ilegg::Int)
    return p.data[isite][ilegg]
end

# Extract ilegg-th index from the projector
#function only(p::Projector, ilegg::Int)::Projector
#data = [[p.data[l][ilegg]] for l in 1:length(p)]
#sitedims = [[p.sitedims[l][ilegg]] for l in 1:length(p)]
#return Projector(data, sitedims)
#end

Base.:(==)(a::Projector, b::Projector)::Bool = (a.data == b.data)
Base.:(<)(a::Projector, b::Projector)::Bool = (a <= b) && (a != b)
Base.:(>)(a::Projector, b::Projector)::Bool = b < a

function Base.:&(a::Projector, b::Projector)::Projector
    a.sitedims == b.sitedims || error("Sitedims mismatch")
    length(a) == length(b) || error("Length mismatch")
    ab = Vector{Int}[]
    for (a_, b_) in zip(a, b)
        ab_ = Int[]
        for (a__, b__) in zip(a_, b_)
            if a__ == 0
                push!(ab_, b__)
            elseif b__ == 0
                push!(ab_, a__)
            elseif a__ == b__
                push!(ab_, a__)
            else
                error("Incompatible projectors $(a) && $(b)")
            end
        end
        push!(ab, ab_)
    end

    return Projector(ab, a.sitedims)
end

function Base.:|(a::Projector, b::Projector)::Projector
    a.sitedims == b.sitedims || error("Sitedims mismatch")
    length(a) == length(b) || error("Length mismatch")
    ab = Vector{Int}[]
    for (a_, b_) in zip(a, b)
        ab_ = Int[]
        for (a__, b__) in zip(a_, b_)
            if a__ == b__
                push!(ab_, a__)
            else
                push!(ab_, 0)
            end
        end
        push!(ab, ab_)
    end

    return Projector(ab, a.sitedims)
end

function Base.:<=(a::Projector, b::Projector)::Bool
    length(a) == length(b) || error("Length mismatch")
    length(a) == length(b) || error("Length mismatch")
    for (a_, b_) in zip(Iterators.flatten(a), Iterators.flatten(b))
        if a_ != 0 && b_ != 0
            if a_ != b_
                return false
            end
        elseif a_ == 0
            if b_ != 0
                return false
            end
        elseif b_ == 0
            # Everything is fine
        end
    end
    return true
end

Base.:>=(a::Projector, b::Projector) = (b <= a)

Base.:<=(a::Vector{Vector{Int}}, b::Projector) = (Projector(a, b.sitedims) <= b)

function hasoverlap(p1, p2)::Bool
    length(p1) == length(p2) || error("Length mismatch")
    for (a, b) in zip(Iterators.flatten(p1), Iterators.flatten(p2))
        if a != 0 && b != 0
            if a != b
                return false
            end
        end
    end
    return true
end

function isleftmmultiidx_contained(p::Projector, leftmmultiidxset::MMultiIndex)::Bool
    _compat(p, i) = (p == 0) || (p == i)
    for n in 1:length(leftmmultiidxset)
        if !all(_compat.(p[n], leftmmultiidxset[n]))
            return false
        end
    end
    return true
end

function isrightmmultiidx_contained(p::Projector, rightmmultiidxset::MMultiIndex)::Bool
    return isleftmmultiidx_contained(reverse(p), reverse(rightmmultiidxset))
end

# QUESTION: How do we reshape in the case of projector with e.g. [2,0]?
function Base.reshape(
    projector::Projector, dims::AbstractVector{<:AbstractVector{Int}}
)::Projector
    length(projector.sitedims) == length(dims) || error("Length mismatch")
    prod.(projector.sitedims) == prod.(dims) || error("Total dimension mismatch")

    newprojectordata = [
        if prod(projector.data[i]) == 0
            zeros(Int, length(dims[i]))
        else
            _multii(dims[i], _lineari(projector.sitedims[i], projector.data[i]))
        end for i in eachindex(projector.data)
    ]
    return Projector(newprojectordata, dims)
end

function isprojectedat(p::Projector, n::Int)::Bool
    if all(p.data[n] .== 0)
        return false
    elseif all(p.data[n] .!= 0)
        return true
    else
        error(
            "Invalid projector $(p.data[n]) at $n, all siteindices at $n must be projected or unprojected",
        )
    end
end

"""
indexset: MMultiIndex, multi indices on unprojected indices
Returns: MMultiIndex, multi indices on all indices

All site indices on each site must be all projected or all unprojected.
"""
# QUESTION: What if they are not all projected or all unprojected? - Gianluca
function fullindices(projector, indexset::MMultiIndex)::MMultiIndex
    sum([prod(projector.data[i]) == 0 for i in eachindex(projector.data)]) == length(indexset) ||
        error("Length mismatch")
    fullidx = Vector{Vector{Int}}(undef, length(projector))
    nsubi = 1
    for n in 1:length(projector)
        if isprojectedat(projector, n)
            fullidx[n] = projector[n]
        else
            fullidx[n] = indexset[nsubi]
            nsubi += 1
        end
    end
    return fullidx
end

fullindices(projector, indexset::MultiIndex)::MultiIndex = lineari(
    projector.sitedims, fullindices(projector, multii(projector.sitedims, indexset))
)

function projectedshape(projector::Projector, startidx::Int, lastidx::Int)::Vector{Int}
    res = Int[
        prod(
            projector[n][s] > 0 ? 1 : projector.sitedims[n][s] for
            s in eachindex(projector.data[n])
        ) for n in startidx:lastidx
    ]
    return res
end
