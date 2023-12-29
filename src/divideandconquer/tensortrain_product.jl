"""
Represents the product of two projected tensor trains
"""
mutable struct ProjectedTensorTrainProduct{T} <: ProjectableEvaluator{T}
    tensortrains::NTuple{2,ProjectedTensorTrain{T,4}} # holds copy of original TTs
    projector::Projector
    sitedims::Vector{Vector{Int}}
    mp::MatrixProduct{T}
end

#function ProjectedTensorTrainProduct(tt::NTuple{2,ProjectedTensorTrain{T,4}}) where {T}
    #return ProjectedTensorTrainProduct{T}(tt)
#end


function create_projected_tensortrain_product(tt::NTuple{2,ProjectedTensorTrain{T,4}}) where {T}
    @show only(tt[1].projector, 2)
    @show only(tt[2].projector, 1)
    if !hasoverlap(only(tt[1].projector, 2), only(tt[2].projector, 1))
        return nothing
    end
    #projector1 = Projector([[x[1], y] for (x, y) in zip(tt[1].projector, pshared)])
    #projector2 = Projector([[x, y[2]] for (x, y) in zip(pshared, tt[2].projector)])
    #project!(tt[1], projector1; compression=false)
    #project!(tt[2], projector2; compression=false)
    projector = Projector([[x[1], y[2]] for (x, y) in zip(tt[1].projector, tt[2].projector)])
    return ProjectedTensorTrainProduct{T}(tt, projector, tt[1].sitedims, MatrixProduct(tt[1].data, tt[2].data))
end

function (obj::ProjectedTensorTrainProduct{T})(indexset::Vector{Int})::T where {T}
    return obj.mp(indexset)
end

function (obj::ProjectedTensorTrainProduct{T})(indexset::Vector{Vector{Int}})::T where {T}
    return obj.mp(indexset)
end

function (obj::ProjectedTensorTrainProduct{T})(
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{T,M + 2} where {T,M}
    if length(leftindexset) * length(rightindexset) == 0
        return zeros(T, 0, 0)
    end
    return obj.mp(leftindexset, rightindexset, Val(M))
end


"""
Collection of the products of two projected tensor trains
An object of this type can be projected to a subset of indices.
"""
mutable struct ProjectedTensorTrainProductSet{T} <: ProjectableEvaluator{T}
    products::OrderedDict{Projector, ProjectedTensorTrainProduct{T}}
    projector::Projector
    sitedims::Vector{Vector{Int}}
end


function ProjectedTensorTrainProductSet(lefttt::AbstractVector{ProjectedTensorTrain{T,4}}, righttt::AbstractVector{ProjectedTensorTrain{T,4}}, projector, sitedims) where {T}
    products = OrderedDict{Projector, ProjectedTensorTrainProduct{T}}()
    for l in lefttt, r in righttt
        p = create_projected_tensortrain_product((l, r))
        if p !== nothing
            products[p.projector] = p
        end
    end
    for (p, v) in products
        p < projector || error("Projector $p is not compatible with $projector")
    end
    return ProjectedTensorTrainProductSet{T}(products, projector, sitedims)
end


#==
function project!(
    obj::ProjectedTensorTrainProduct{T},
    prj::AbstractVector{<:AbstractVector{Int}};
    compression::Bool=false,
    cutoff::Float64=1e-30,
    maxdim::Int=typemax(Int)
)::ProjectedTensorTrainProduct{T} where {T}
    all([length(p) == 2 for p in prj]) || error("Number of siteindices must be 2!")

    # TODO: compatibility

    p1 = [[p[1][1], p[2][2]] for p in zip(prj, obj.projector)]
    p2 = [[p[1][1], p[2][2]] for p in zip(obj.projector, prj)]
    project!(obj.tensortrain[1], p1; compression=compression, cutoff=cutoff, maxdim=maxdim)
    project!(obj.tensortrain[2], p2; compression=compression, cutoff=cutoff, maxdim=maxdim)

    return obj
end
==#