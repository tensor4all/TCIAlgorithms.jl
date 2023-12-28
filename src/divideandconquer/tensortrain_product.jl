mutable struct ProjectedTensorTrainProduct{T} <: ProjectableEvaluator{T}
    tensortrains::NTuple{2,ProjectedTensorTrain{T,4}} # holds copy of original TTs
    projector::Projector
    sitedims::Vector{Vector{Int}}
    mp::MatrixProduct{T}
end

function ProjectedTensorTrainProduct(tt::NTuple{2,ProjectedTensorTrain{T,4}}) where {T}
    return ProjectedTensorTrainProduct{T}(tt)
end

function ProjectedTensorTrainProduct{T}(tt::NTuple{2,ProjectedTensorTrain{T,4}}) where {T}
    pshared = Int[]
    for l in 1:length(tt[1])
        p1_ = tt[1].projector[l][2]
        p2_ = tt[2].projector[l][1]
        if p1_ == p2_
            push!(pshared, p1_)
        else
            # here, p1_ != p2_
            if p1_ == 0
                push!(pshared, p2_)
            elseif p2_ == 0
                push!(pshared, p1_)
            else
                @assert p1_ != 0 && p2_ != 0
                if p1_ == p2_
                    push!(pshared, p1_)
                else
                    push!(pshared, -1)
                end
            end
        end
    end

    if findfirst(x -> x == -1, pshared) !== nothing
        return nothing
    end

    projector1 = Projector([[x[1], y] for (x, y) in zip(tt[1].projector, pshared)])
    projector2 = Projector([[x, y[2]] for (x, y) in zip(pshared, tt[2].projector)])

    project!(tt[1], projector1; compression=true)
    project!(tt[2], projector2; compression=true)

    projector = Projector([[x[1], y[2]] for (x, y) in zip(tt[1].projector, tt[2].projector)])

    return ProjectedTensorTrainProduct{T}(tt, projector, tt[1].sitedims, MatrixProduct(tt[1].data, tt[2].data))
end

function (obj::ProjectedTensorTrainProduct{T})(indexset::Vector{Int})::T where {T}
    return obj.mp(indexset)
end

#function (obj::ProjectedTensorTrainProduct{T})(indexset::Vector{Vector{Int}})::T where {T}
    #return obj.mp(indexset)
#end

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