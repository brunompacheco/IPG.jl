
const AssignmentDict = Dict{VariableRef,Float64}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{Float64}) = AssignmentDict(zip(all_variables(player), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment

"Replace the variables in an expression with their assigned values."
function replace(expr::AbstractJuMPScalar, assignment::AssignmentDict)::AbstractJuMPScalar
    function _recursive_replace(expr::AbstractJuMPScalar)::AbstractJuMPScalar
        if expr isa VariableRef
            return get(assignment, expr, expr)
        elseif expr isa AffExpr
            # TODO: this behavior of value is unintended and may lead to problems in the future
            return value(v -> get(assignment, v, v), expr)
        elseif expr isa QuadExpr
            return value(v -> get(assignment, v, v), expr)
        elseif expr isa NonlinearExpr
            return NonlinearExpr(expr.head, map(_recursive_replace, expr.args))
        else
            return expr
        end
    end

    return _recursive_replace(expr)
end
