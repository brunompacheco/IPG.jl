
"Compute `player`'s best response to the mixed strategy profile `σ_others`."
function best_response(player::Player, σ_others::Profile{DiscreteMixedStrategy})
    vars_player = all_variables(player)

    obj = payoff(player, vars_player, σ_others)

    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.X, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.X, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.X)
    optimize!(player.X)

    return value.(vars_player)
end
"Compute `player`'s best response to the pure strategy profile `x_others`."
best_response(player::Player, x_others::Profile{PureStrategy}) = best_response(player, convert(Profile{DiscreteMixedStrategy}, x_others))
