
using OrderedCollections: OrderedDict

const Sample{T<:Strategy} = OrderedDict{Player, Vector{T}}  # sample of strategies for each player
Sample(ps::Pair{Player, Vector{T}}...) where T<:Strategy = Sample{T}(ps...)
export Sample

abstract type AbstractSampledGame end

include("PolymatrixSampledGame/PolymatrixSampledGame.jl")

include("UtilitiesSampledGame/UtilitiesSampledGame.jl")

function SampledGame(S_X::Sample{PureStrategy})
    players = collect(keys(S_X))

    if length(players) == 2
        SampledGameType = PolymatrixSampledGame
    # TODO: detect separable payoffs. Also, the following might miss constant payoffs, although that's a bit bizarre
    elseif all(p.Π isa Union{QuadExpr,AffExpr})
        SampledGameType = PolymatrixSampledGame
    else
        println("WARNING: Separable payoff was NOT detected. Using utility-based sampled game.")
        SampledGameType = UtilitiesSampledGame
    end

    return SampledGameType(S_X)
end
