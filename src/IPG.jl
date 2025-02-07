module IPG

using NormalGames, JuMP

include("Strategies.jl")
include("Payoff.jl")
include("Player.jl")
include("SGM.jl")

export Player, QuadraticPayoff, payoff, DiscreteMixedStrategy, expected_value, @variable, @constraint

end # module IPG
