var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = TCIAlgorithms","category":"page"},{"location":"#TCIAlgorithms","page":"Home","title":"TCIAlgorithms","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for TCIAlgorithms.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [TCIAlgorithms]","category":"page"},{"location":"#TCIAlgorithms.AbstractPatchCreator","page":"Home","title":"TCIAlgorithms.AbstractPatchCreator","text":"T: Float64, ComplexF64, etc. M: TensorCI2, MPS, etc.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.ElementwiseProduct","page":"Home","title":"TCIAlgorithms.ElementwiseProduct","text":"Elementwise product of two tensor trains One site index on each site.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.MatrixProduct","page":"Home","title":"TCIAlgorithms.MatrixProduct","text":"Matrix product of two tensor trains Two site indices on each site.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.MatrixProductSum","page":"Home","title":"TCIAlgorithms.MatrixProductSum","text":"Sum of matrix products of two tensor trains Two site indices on each site.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.PartitionedTensorTrain","page":"Home","title":"TCIAlgorithms.PartitionedTensorTrain","text":"Collection of ProjectableEvaluator objects\n\nThe underlying data will be copied when projected.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.PatchOrdering","page":"Home","title":"TCIAlgorithms.PatchOrdering","text":"Specify the ordering of patching\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.ProjectableEvaluator","page":"Home","title":"TCIAlgorithms.ProjectableEvaluator","text":"Type for an object that can be projected on a subset of indices\n\nAttributes:\n\nprojector: Projector object\nsitedims: Vector{Vector{Int}} of the dimensions of the local indices\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.ProjectedTensorTrain","page":"Home","title":"TCIAlgorithms.ProjectedTensorTrain","text":"TensorTrain projected on a subset of indices\n\nThe underlying data will be copied when projected.\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.ProjectedTensorTrain-Union{Tuple{T}, Tuple{TensorCrossInterpolation.TensorTrain{T, 3}, AbstractVector{<:AbstractVector{Int64}}, TCIAlgorithms.Projector}} where T","page":"Home","title":"TCIAlgorithms.ProjectedTensorTrain","text":"tt is a TensorTrain{T,3} and prj is a Projector. tt is defined on unprojected indices. Return a ProjectedTensorTrain{T} defined on full indices.\n\n\n\n\n\n","category":"method"},{"location":"#TCIAlgorithms.ProjectedTensorTrainProduct","page":"Home","title":"TCIAlgorithms.ProjectedTensorTrainProduct","text":"Represents the product of two projected tensor trains\n\n\n\n\n\n","category":"type"},{"location":"#TCIAlgorithms.deltaproduct-Union{Tuple{T}, Tuple{AbstractMatrix{T}, AbstractMatrix{T}}} where T","page":"Home","title":"TCIAlgorithms.deltaproduct","text":"Elementwise product in one index: C_ijk = A_ij B_jk\n\n\n\n\n\n","category":"method"},{"location":"#TCIAlgorithms.maskactiveindices-Tuple{TCIAlgorithms.PatchOrdering, Int64}","page":"Home","title":"TCIAlgorithms.maskactiveindices","text":"n is the length of the prefix.\n\n\n\n\n\n","category":"method"},{"location":"#TCIAlgorithms.sum-Union{Tuple{TCIAlgorithms.PartitionedTensorTrain{T}}, Tuple{T}} where T","page":"Home","title":"TCIAlgorithms.sum","text":"Sum over external indices\n\n\n\n\n\n","category":"method"}]
}