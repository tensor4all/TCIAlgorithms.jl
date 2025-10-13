module TCIAlgorithms

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation:
    TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2

using ITensors
using ITensorMPS: ITensorMPS
using ITensorMPS: MPO, MPS, linkdims, linkinds
using Quantics

using OrderedCollections: OrderedDict, OrderedSet
using Distributed
using EllipsisNotation

import TensorCrossInterpolation as TCI
import TensorCrossInterpolation:
    TensorTrain, evaluate, TTCache, MultiIndex, LocalIndex, TensorCI2
import LinearAlgebra as LA

import FastMPOContractions as FMPOC

const MMultiIndex = Vector{Vector{Int}}
const TensorTrainState{T} = TensorTrain{T,3} where {T}

include("util.jl")
include("projector.jl")
include("blockstructure.jl")
include("projectable_evaluator.jl")
include("projtensortrain.jl")
include("container.jl")
include("mul.jl")
include("distribute.jl")
include("tree.jl")
include("patching.jl")
include("crossinterpolate.jl")
include("adaptivematmul.jl")

# ITensor interface
include("itensor.jl")

end
