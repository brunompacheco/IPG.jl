module IPG

using NormalGames, JuMP

include("Game/Strategies.jl")
include("Game/Payoff.jl")
include("Game/Player.jl")
include("SGM/SampledGame.jl")
include("SGM/SGM.jl")

export Player, QuadraticPayoff, payoff, DiscreteMixedStrategy, expected_value, @variable, @constraint

end # module IPG
