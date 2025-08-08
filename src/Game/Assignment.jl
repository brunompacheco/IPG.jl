
const AssignmentDict = Dict{VariableRef,Float64}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{Float64}) = AssignmentDict(zip(all_variables(player), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment
