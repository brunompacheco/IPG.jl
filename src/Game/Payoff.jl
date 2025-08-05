
"""
Get the payoff map for `player` given the pure strategy profile `x_others`.
The payoff map is a function that takes the player's strategy and returns the payoff.
"""
function get_payoff_map(player::Player, x_others::Profile{<:Vector{<:Real}})
    variable_assignments = Dict{VariableRef, Any}()
    for (other, x_other) in x_others
        for (v, val) in zip(all_variables(other.X), x_other)
            variable_assignments[v] = val
        end
    end

    function payoff_map(x_player::Vector{<:Any})
        # Assign the player's strategy to the variables
        for (v, val) in zip(all_variables(player.X), x_player)
            variable_assignments[v] = val
        end

        return value(v -> get(variable_assignments, v, 0), player.Π)
    end

    return payoff_map
end

"Evaluate the player's payoff when she plays `x_player` and the others play `x_others`."
function payoff(player::Player, x_player::Vector{<:Any}, x_others::Profile{<:Vector{<:Real}})
    return get_payoff_map(player, x_others)(x_player)
end

"Expected payoff of a pure strategy (`x_player`) against a mixed profile (`σ_others`)."
function payoff(player::Player, x_player::Vector{<:Any}, σ_others::Profile{DiscreteMixedStrategy})
    return expected_value(x_others -> payoff(player, x_player, x_others), σ_others)
end

"Expected payoff of a mixed strategy (`σ_player`) against a pure profile (`x_others`)."
function payoff(player::Player, σ_player::DiscreteMixedStrategy, x_others::Profile{<:Vector{<:Real}})
    return expected_value(x_player -> payoff(player, x_player, x_others), σ_player)
end

"Expected payoff of a mixed strategy (`σ_player`) against a mixed profile (`σ_others`)."
function payoff(player::Player, σ_player::DiscreteMixedStrategy, σ_others::Profile{DiscreteMixedStrategy})
    return expected_value(x_player -> payoff(player, x_player, σ_others), σ_player)
end
