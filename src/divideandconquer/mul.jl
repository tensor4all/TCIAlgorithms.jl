istto(obj::ProjectableEvaluator)::Bool = all([length(x) == 2 for x in obj.projector])

function create_multiplier(
    ptt1::PartitionedTensorTrain{T}, ptt2::PartitionedTensorTrain{T}
)::PartitionedTensorTrain{T} where {T}
    istto(ptt1) || error("ptt1 is not a TTO")
    istto(ptt2) || error("ptt2 is not a TTO")

    globalprojector = Projector(
        [[x[1], y[2]] for (x, y) in zip(ptt1.projector, ptt2.projector)],
        [[x[1], y[2]] for (x, y) in zip(ptt1.sitedims, ptt2.sitedims)],
    )
    return create_multiplier(
        Vector{ProjectedTensorTrain{T,4}}(ptt1.tensortrains),
        Vector{ProjectedTensorTrain{T,4}}(ptt2.tensortrains),
        globalprojector,
    )
end


function contract_tto(
    ptt1::PartitionedTensorTrain{T},
    ptt2::PartitionedTensorTrain{T};
    maxbonddim=typemax(Int),
    rtol=1e-10,
    verbosity=0,
    ntry=10,
    loginterval=1,
    ninitialpivot=100,
    initialpivots=MultiIndex[],
    patchordering=PatchOrdering(collect(1:length(ptt1.sitedims))),
)::PartitionedTensorTrain{T} where {T}
    istto(ptt1) || error("ptt1 is not a TTO")
    istto(ptt2) || error("ptt2 is not a TTO")

    sitedims = [[x[1] * y[2]] for (x, y) in zip(ptt1.sitedims, ptt2.sitedims)]
    localdims = [x[1] * y[2] for (x, y) in zip(ptt1.sitedims, ptt2.sitedims)]
    creator = TCI2PatchCreator(
        T,
        create_multiplier(ptt1, ptt2),
        localdims;
        maxbonddim=maxbonddim,
        rtol=rtol,
        verbosity=verbosity,
        ntry=ntry,
        loginterval=loginterval,
        ninitialpivot=ninitialpivot,
        initialpivots=initialpivots,
    )
    tree = adaptiveinterpolate(
        creator, patchordering; verbosity=verbosity, maxnleaves=typemax(Int)
    )
    return PartitionedTensorTrain(tree, sitedims, patchordering)
end