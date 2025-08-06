
include("Polymatrix.jl")

"Normal-form polymatrix representation of the sampled game."
mutable struct PolymatrixSampledGame
    S_X::Dict{Player, Vector{PureStrategy}}  # sample of strategies (finite subset of the strategy space X)
    polymatrix::Polymatrix
end

# TODO: since S_X is a Dict{Player, ...}, we don't need to pass the players anymore
function PolymatrixSampledGame(players::Vector{Player}, S_X::Dict{Player, Vector{PureStrategy}})
    return PolymatrixSampledGame(S_X, get_polymatrix(players, S_X))
end

# TODO: maybe each Strategy should have a pointer to the player it belongs to?
function add_new_strategy!(sg::PolymatrixSampledGame, p::Player, new_xp::PureStrategy)
    # first part is easy, just add the new strategy to the set
    push!(sg.S_X[p], new_xp)

    update_polymatrix!(sg.polymatrix, p, sg.S_X)
end

include("Solve.jl")
