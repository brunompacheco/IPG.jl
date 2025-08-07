
#TODO: maybe we should handle VariableRef to VariableRef mappings outside of assignments,
# i.e., Assignment = Dict{VariableRef,Float64}
const AssignmentDict = Dict{VariableRef,Any}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{<:Any}) = AssignmentDict(zip(all_variables(player.X), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment
