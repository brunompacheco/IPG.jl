module IPG

using JuMP
export @variable, @constraint

include("Game/Game.jl")
include("SGM/SGM.jl")

end # module IPG
