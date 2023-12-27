mutable struct ProjectedTensorTrainProduct{T} <: ProjectableEvaluator{T}
    tensortrain::NTuple{2,ProjectedTensorTrain{T,4}}
    projector::Vector{Vector{Int}}
    sitedims::Vector{Vector{Int}}
end

function ProjectedTensorTrainProduct{T}(tt::NTuple{2,ProjectedTensorTrain{T,4}}) where {T}
    p1 = projector(tt[1], 1)
    p2 = projector(tt[2], 2)
    s1 = sitedims(tt[1], 1)
    s2 = sitedims(tt[2], 2)
    return ProjectedTensorTrainProduct{T}(tt,
        [collect(p) for p in zip(p1, p2)],
        [collect(s) for s in zip(s1, s2)])
end

function partition!(
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
    partition!(obj.tensortrain[1], p1; compression=compression, cutoff=cutoff, maxdim=maxdim)
    partition!(obj.tensortrain[2], p2; compression=compression, cutoff=cutoff, maxdim=maxdim)

    return obj
end
