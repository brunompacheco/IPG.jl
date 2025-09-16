
using OrderedCollections: OrderedDict

const Sample{T<:Strategy} = OrderedDict{Player, Vector{T}}  # sample of strategies for each player
Sample(ps::Pair{Player, Vector{T}}...) where T<:Strategy = Sample{T}(ps...)
export Sample

abstract type AbstractSampledGame end

include("PolymatrixSampledGame/PolymatrixSampledGame.jl")

include("UtilitiesSampledGame/UtilitiesSampledGame.jl")

solve = nothing  # placeholder

function SampledGame(S_X::Sample{PureStrategy})
    players = collect(keys(S_X))

    # TODO: detect separable payoffs. Also, the following might miss constant payoffs,
    # although that'd be a bizarre case
    if (length(players) == 2) || all(p.Π isa Union{QuadExpr,AffExpr} for p in players)
        SampledGameType = PolymatrixSampledGame
        global solve = solve_polymatrix_game
    else
        println("WARNING: Separable payoff was NOT detected. Using utility-based sampled game.")
        SampledGameType = UtilitiesSampledGame
        global solve = solve_utilities_game
    end

    return SampledGameType(S_X)
end
