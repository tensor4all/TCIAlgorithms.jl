"""
Represents the product of two projected tensor trains
"""
mutable struct ProjectedTensorTrainProduct{T} <: ProjectableEvaluator{T}
    tensortrains::NTuple{2,ProjectedTensorTrain{T,4}}
    projector::Projector
    sitedims::Vector{Vector{Int}}
    mp::MatrixProduct{T}
end

function create_projected_tensortrain_product(
    tt::NTuple{2,ProjectedTensorTrain{T,4}}
) where {T}
    if !hasoverlap(only(tt[1].projector, 2), only(tt[2].projector, 1))
        return nothing
    end
    projector = Projector([
        [x[1], y[2]] for (x, y) in zip(tt[1].projector, tt[2].projector)
    ])
    return ProjectedTensorTrainProduct{T}(
        tt, projector, tt[1].sitedims, MatrixProduct(tt[1].data, tt[2].data)
    )
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
#mutable struct TensorTrainMutiplier{T} <: ProjectableEvaluator{T}
#products::OrderedSet{ProjectedTensorTrainProduct{T}}
#projector::Projector
#sitedims::Vector{Vector{Int}}
#end


function create_multiplier(
    lefttt::AbstractVector{ProjectedTensorTrain{T,4}},
    righttt::AbstractVector{ProjectedTensorTrain{T,4}},
    projector,
) where {T}
    sitedims = [[x[1], y[2]] for (x, y) in zip(lefttt[1].sitedims, righttt[1].sitedims)]

    products = Vector{ProjectedTensorTrainProduct{T}}()
    for l in lefttt, r in righttt
        p = create_projected_tensortrain_product((l, r))
        if p !== nothing
            push!(products, p)
        end
    end
    for v in products
        v.projector <= projector ||
            error("Projector $(v.projector) is not compatible with $projector")
    end
    return PartitionedTensorTrain(products, projector, sitedims)
end