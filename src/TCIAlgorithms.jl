module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation: TensorTrain, evaluate, TTCache, MultiIndex

using ITensors
import ITensorTDVP

include("tensor.jl")
include("matrixmulalgorithms/fitalgorithm.jl")
include("matrixmul.jl")
include("matrixmulsum.jl")
include("elementwisemul.jl")

end
