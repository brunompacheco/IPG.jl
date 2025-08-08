
#TODO: maybe we should handle VariableRef to VariableRef mappings outside of assignments,
# i.e., Assignment = Dict{VariableRef,Float64}
const AssignmentDict = Dict{VariableRef,Any}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{<:Any}) = AssignmentDict(zip(all_variables(player.X), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment

function replace_in_expression(expr::AbstractJuMPScalar, x::AssignmentDict)
    function _recursive_replace(expr::AbstractJuMPScalar)::AbstractJuMPScalar
        if expr isa VariableRef
            if haskey(x, expr)
                return x[expr]
            else
                return expr
            end
        elseif expr isa AffExpr
            # TODO: the following use of value is accidental and can break in the future.
            return value(v -> haskey(x, v) ? x[v] : v, expr)
        elseif expr isa QuadExpr
            return value(v -> haskey(x, v) ? x[v] : v, expr)
        elseif expr isa NonlinearExpr
            # replaced_expr = NonlinearExpr(expr.head, Any[_recursive_replace(arg) for arg in expr.args]...)
            # replaced_expr = NonlinearExpr(expr.head, map(_recursive_replace, expr.args))
            # if ~any(arg isa NonlinearExpr for arg in expr.args)
            #     replaced_expr =
            #     g = MOI.Nonlinear.SymbolicAD.simplify(moi_function(f))
            #     replaced_expr = jump_function(model, g)

            # end
            # return 
            error("Nonlinear expressions are not supported in IPG yet.")
        else
            return expr
        end
    end

    return _recursive_replace(expr)
end
