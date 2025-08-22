
include("Utilities.jl")

mutable struct UtilitiesSampledGame <: AbstractSampledGame
    "Sample of strategies (finite subset of the strategy space X)."
    S_X::Sample{PureStrategy}
    "Utilities of each player for each possible profile in the game."
    utilities::Array{Float64}  # n+1 dimensional: (s₁, ..., sₙ, n)
end
function UtilitiesSampledGame(players::Vector{Player}, S_X::Sample{PureStrategy})
    return UtilitiesSampledGame(S_X, get_utilities(players, S_X))
end
