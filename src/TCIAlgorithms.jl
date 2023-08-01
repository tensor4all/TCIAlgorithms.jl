module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex

include("tensor.jl")
#include("mpo.jl")
include("matrixmul.jl")
include("elementwisemul.jl")

end
