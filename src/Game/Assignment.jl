
#TODO: maybe we should handle VariableRef to VariableRef mappings outside of assignments,
# i.e., Assignment = Dict{VariableRef,Float64}
const AssignmentDict = Dict{VariableRef,Any}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{<:Any}) = AssignmentDict(zip(all_variables(player.X), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment

function simplify_expression(expr::AbstractJuMPScalar, x::AssignmentDict)
    function _recursive_simplify_expr(expr::AbstractJuMPScalar)::AbstractJuMPScalar
        if expr isa VariableRef
            if haskey(x, expr)
                return x[expr]
            else
                return expr
            end
        elseif expr isa AffExpr
            return value(v -> haskey(x, v) ? x[v] : v, expr)
        elseif expr isa QuadExpr
            return value(v -> haskey(x, v) ? x[v] : v, expr)
        elseif expr isa NonlinearExpr
            error("Nonlinear expressions are not supported in IPG yet.")
        else
            error("Unknown expression type: $(typeof(expr))")
        end
    end

    return _recursive_simplify_expr(expr)
end
