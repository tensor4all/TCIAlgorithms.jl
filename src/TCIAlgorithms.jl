module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2

using TCIITensorConversion
using ITensors
import ITensorTDVP

using OrderedCollections: OrderedDict, OrderedSet

using Distributed

using EllipsisNotation

include("util.jl")
include("tensor.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")

include("divideandconquer/abstracttypes.jl")
include("divideandconquer/projected_tensortrain.jl")
include("divideandconquer/tensortrain_product.jl")
include("divideandconquer/partitioned_tensortrain.jl")
include("divideandconquer/adaptivepartitioning.jl")
#include("divideandconquer/tensortrain_product.jl")

end
