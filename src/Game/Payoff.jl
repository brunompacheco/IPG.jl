
"""
Create a dictionary of variable assignments (JuMP-style) from a pure strategy.
"""
function build_var_assignments(player::Player, x::Vector{<:Any})
    variable_assignments = Dict{VariableRef, Any}()
    for (v, val) in zip(all_variables(player.X), x)
        variable_assignments[v] = val
    end
    return variable_assignments
end

"""
Get the payoff map for `player` given the pure strategy profile `x_others`.
The payoff map is a function that takes the player's strategy and returns the payoff.
"""
function get_payoff_map(player::Player, x_others::Profile{PureStrategy})
    others_var_assignments = [build_var_assignments(other, x_other)
                              for (other, x_other) in x_others]
    var_assignments = merge(others_var_assignments...)

    function payoff_map(x_player::Vector{<:Any})
        complete_var_assignments = merge(var_assignments, build_var_assignments(player, x_player))

        return value(v -> complete_var_assignments[v], player.Π)
    end

    return payoff_map
end

"Evaluate the player's payoff when she plays `x_player` and the others play `x_others`."
function payoff(player::Player, x_player::Vector{<:Any}, x_others::Profile{PureStrategy})
    return get_payoff_map(player, x_others)(x_player)
end

"Expected payoff of a pure strategy (`x_player`) against a mixed profile (`σ_others`)."
function payoff(player::Player, x_player::Vector{<:Any}, σ_others::Profile{DiscreteMixedStrategy})
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
