module IPG

using NormalGames

include("Strategies.jl")
include("Player.jl")

export Player, QuadraticPayoff, payoff, DiscreteMixedStrategy, expected_value

end # module IPG
