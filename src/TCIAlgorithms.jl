module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex

using ITensors

include("tensor.jl")
#include("mpo.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")

end
