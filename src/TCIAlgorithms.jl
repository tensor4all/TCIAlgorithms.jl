module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex

using TCIITensorConversion

using ITensors
import ITensorTDVP

include("tensor.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")
include("partitioned_tensortrain.jl")

end
