
const Sample{T<:Strategy} = Dict{Player, Vector{T}}  # sample of strategies for each player

abstract type AbstractSampledGame end

include("PolymatrixSampledGame/PolymatrixSampledGame.jl")

include("UtilitiesSampledGame/UtilitiesSampledGame.jl")
