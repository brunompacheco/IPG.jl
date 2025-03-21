module IPG

using JuMP

include("Game/Strategies.jl")
include("Game/Payoff/Payoff.jl")
include("Game/Player.jl")
include("SGM/SGM.jl")

export Player, QuadraticPayoff, BlackBoxPayoff, payoff, DiscreteMixedStrategy, expected_value, @variable, @constraint, others, AbstractPlayer

end # module IPG
