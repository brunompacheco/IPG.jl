
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

function replace_in_payoff(player::Player, assignment::AssignmentDict)::AbstractJuMPScalar
    internal_assignment = _internalize_assignment(player, assignment)
    return replace(player.Π, internal_assignment)
end

"""
Get the payoff map for `player` given the pure strategy profile `x_others`.
The payoff map is a function that takes the player's strategy and returns the payoff.
"""
function get_payoff_map(player::Player, x_others::Profile{PureStrategy})
    assignment_others = Assignment(x_others)
    internal_assignment_others = _internalize_assignment(player, assignment_others)

    function payoff_map(x_player::Vector{Float64})
        complete_assignment = merge(internal_assignment_others, Assignment(player, x_player))
        return value(v -> complete_assignment[v], player.Π)
    end

    return payoff_map
end

"Evaluate the player's payoff when she plays `x_player` and the others play `x_others`."
function payoff(player::Player, x_player::PureStrategy, x_others::Profile{PureStrategy})
    return get_payoff_map(player, x_others)(x_player)
end

"Expected payoff of a pure strategy (`x_player`) against a mixed profile (`σ_others`)."
function payoff(player::Player, x_player::PureStrategy, σ_others::Profile{DiscreteMixedStrategy})
    return expected_value(x_others -> payoff(player, x_player, x_others), σ_others)
end

"Expected payoff of a mixed strategy (`σ_player`) against a pure profile (`x_others`)."
function payoff(player::Player, σ_player::DiscreteMixedStrategy, x_others::Profile{PureStrategy})
    return expected_value(x_player -> payoff(player, x_player, x_others), σ_player)
end

"Expected payoff of a mixed strategy (`σ_player`) against a mixed profile (`σ_others`)."
function payoff(player::Player, σ_player::DiscreteMixedStrategy, σ_others::Profile{DiscreteMixedStrategy})
    return expected_value(x_player -> payoff(player, x_player, σ_others), σ_player)
end

payoff(player::Player, σ::Profile{T}) where T <: Strategy = payoff(player, σ[player], others(σ, player))

export payoff
