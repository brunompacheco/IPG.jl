
using OrderedCollections: OrderedDict

const Sample{T<:Strategy} = OrderedDict{Player, Vector{T}}  # sample of strategies for each player
Sample(ps::Pair{Player, Vector{T}}...) where T<:Strategy = Sample{T}(ps...)
export Sample

abstract type AbstractSampledGame end

include("PolymatrixSampledGame/PolymatrixSampledGame.jl")

include("UtilitiesSampledGame/UtilitiesSampledGame.jl")
