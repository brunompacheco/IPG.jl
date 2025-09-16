
include("Utilities.jl")

mutable struct UtilitiesSampledGame <: AbstractSampledGame
    "Sample of strategies (finite subset of the strategy space X)."
    S_X::Sample{PureStrategy}
    "Utilities of each player for each possible profile in the game."
    utilities::Array{Float64}  # n+1 dimensional: (s₁, ..., sₙ, n)
end
function UtilitiesSampledGame(S_X::Sample{PureStrategy})
    return UtilitiesSampledGame(S_X, get_utilities(S_X))
end

function add_new_strategy!(sg::UtilitiesSampledGame, p::Player, new_xp::PureStrategy)
    # first part is easy, just add the new strategy to the set
    push!(sg.S_X[p], new_xp)

    sg.utilities = update_utilities(sg.utilities, sg.S_X)
end

include("Solve.jl")
