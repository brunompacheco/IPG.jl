
const AssignmentDict = Dict{VariableRef,Float64}

"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
Assignment(player::Player, x::PureStrategy) = AssignmentDict(zip(all_variables(player), x))
Assignment(x::Profile{PureStrategy})::AssignmentDict = merge(collect(Assignment(p, x_p) for (p, x_p) in x)...)
export Assignment

"Replace the variables in an expression with their assigned values."
function replace(expr::AbstractJuMPScalar, assignment::AssignmentDict)
    _recursive_replace(expr::Number) = expr
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
            if !any(arg isa NonlinearExpr for arg in replaced_expr.args)
                # TODO: the following is to be replaced by JuMP.simplify once https://github.com/jump-dev/JuMP.jl/pull/4047 gets merged
                g = MOI.Nonlinear.SymbolicAD.simplify(moi_function(replaced_expr))

                # TODO: this owner model assignment is really ugly. maybe the owner model should be an argument
                terms_in_replaced_expr = filter(v -> v isa JuMP.AbstractJuMPScalar, replaced_expr.args)
                owner = nothing
                for term in terms_in_replaced_expr
                    term_owner = owner_model(term)
                    if ~isnothing(term_owner)
                        owner = term_owner
                        break
                    end
                end

                if isnothing(owner)
                    # Explicitly handle the case where g is not a JuMP expression
                    if g isa Number
                        replaced_expr = g
                    else
                        error("replace: Unable to convert simplified expression to a number or JuMP expression. Got: $(typeof(g))")
                    end
                else
                    # update owner model
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

"Translate variable references of the assignment to internal references."
function _internalize_assignment(player::Player, assignment::AssignmentDict)
    internal_assignment = AssignmentDict()
    for (v_ref, v_val) in assignment
        if v_ref ∈ all_variables(player.X)
            internal_assignment[v_ref] = v_val
        elseif v_ref ∈ keys(player._param_dict)
            internal_assignment[player._param_dict[v_ref]] = v_val
        end
    end

    return internal_assignment
end
