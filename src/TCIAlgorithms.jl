module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation:
    TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2

using TCIITensorConversion
using ITensors

using OrderedCollections: OrderedDict, OrderedSet
using Distributed
using EllipsisNotation

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation:
    TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2
using TCIITensorConversion
import LinearAlgebra as LA

using ITensors # TO BE REMOVED

const MMultiIndex = Vector{Vector{Int}}

include("util.jl")
include("tensor.jl")
include("adapter.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")

include("divideandconquer/projector.jl")
include("divideandconquer/adapter.jl")
include("divideandconquer/tensortrain.jl")
include("divideandconquer/projected_tensortrain.jl")
include("divideandconquer/tensortrain_product.jl")
include("divideandconquer/partitioned_tensortrain.jl")
include("divideandconquer/adaptivepartitioning.jl")
#include("divideandconquer/tensortrain_product.jl")

end
