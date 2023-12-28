module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex

using TCIITensorConversion

using ITensors
import ITensorTDVP

#using OrderedCollections: OrederedDict

include("util.jl")
include("tensor.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")

include("divideandconquer/abstracttypes.jl")
include("divideandconquer/projected_tensortrain.jl")
include("divideandconquer/tensortrain_product.jl")
include("divideandconquer/partitioned_tensortrain.jl")
#include("divideandconquer/tensortrain_product.jl")

end
