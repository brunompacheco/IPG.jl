module IPG

using JuMP

include("Game/Game.jl")
include("SGM/SGM.jl")

export Player, QuadraticPayoff, BlackBoxPayoff, payoff, DiscreteMixedStrategy, expected_value, @variable, @constraint, others, AbstractPlayer

end # module IPG
