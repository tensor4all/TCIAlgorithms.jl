#struct PartitionedTensorTrain{T,N} <: ProjectableEvaluator{T}
    #tensortrains::ProjectedTensorTrain{T,N}
    #projector::Vector{Vector{Int}} # (L, N-2)
    #sitedims::Vector{Vector{Int}} # (L, N-2)
#end
