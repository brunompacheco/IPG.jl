module IPG

using JuMP
export @variable, @constraint

# TODO: this is temporary, while MultiPlayer is not made public
println("WARNING: MultiPlayer is being imported via direct inclusion of the source files.")
include("../MultiPlayer/src/MultiGame.jl")
include("../MultiPlayer/src/NashAlgorithms.jl")

include("Game/Game.jl")
include("SGM/SGM.jl")

end # module IPG
