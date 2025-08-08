
const AssignmentDict = Dict{VariableRef,Float64}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::Vector{Float64}) = AssignmentDict(zip(all_variables(player), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment

"Replace the variables in an expression with their assigned values."
function replace(expr::AbstractJuMPScalar, assignment::AssignmentDict)
    function _recursive_replace(expr::AbstractJuMPScalar)
        if expr isa VariableRef
            return get(assignment, expr, expr)
        elseif expr isa AffExpr
            # TODO: this behavior of value is unintended and may lead to problems in the future
            return value(v -> get(assignment, v, v), expr)
        elseif expr isa QuadExpr
            return value(v -> get(assignment, v, v), expr)
        elseif expr isa NonlinearExpr
            replaced_expr = NonlinearExpr(expr.head, Vector{Any}(map(_recursive_replace, expr.args)))
            # If there are no nonlinear arguments, we can try to simplify the resulting expression
            if ~any(arg isa NonlinearExpr for arg in replaced_expr.args)
                #TODO: the following is to be replaced by JuMP.simplify once https://github.com/jump-dev/JuMP.jl/pull/4047 gets merged
                g = MOI.Nonlinear.SymbolicAD.simplify(moi_function(replaced_expr))

                terms_in_replaced_expr = filter(v -> v isa JuMP.AbstractJuMPScalar, replaced_expr.args)

                if length(terms_in_replaced_expr) == 0
                    replaced_expr = g  # the result is likely just a number
                else
                    # update owner model
                    owner = owner_model(first(terms_in_replaced_expr))

                    replaced_expr = jump_function(owner, g)
                end
            end
            return replaced_expr
        else
            return expr
        end
    end

    return _recursive_replace(expr)
end
