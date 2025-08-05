
"Compute `player`'s best response to the pure strategy profile `x_others`."
function best_response(player::Player, x_others::Profile{<:Vector{<:Real}})
    dummy_mixed_profile = Profile{DiscreteMixedStrategy}(
        p => DiscreteMixedStrategy(xp) for (p, xp) in x_others
    )
    return best_response(player, dummy_mixed_profile)
end

"Compute `player`'s best response to the mixed strategy profile `σ_others`."
function best_response(player::Player, σ_others::Profile{DiscreteMixedStrategy})
    vars_player = all_variables(player.X)

    obj = payoff(player, vars_player, σ_others)

    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.X, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.X, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.X)
    optimize!(player.X)

    return value.(vars_player)
end
# function best_response(player::Player{<:AbstractBilateralPayoff}, σ::Vector{DiscreteMixedStrategy})
    # error("best_response for player with bilateral payoff not implemented yet") # TODO

    # xp = all_variables(player.X)

    # # TODO: No idea why this doesn't work
    # # @objective(model, Max, sum([IPG.bilateral_payoff(Πp, p, xp, k, σ[k]) for k in 1:m]))

    # obj = AffExpr()
    # for k in eachindex(σ)
    #     if k == player.p
    #         obj += IPG.bilateral_payoff(player.Π, xp)
    #     else
    #         obj += IPG.bilateral_payoff(player.Π, xp, σ[k], k)
    #     end
    # end
    # # I don't know why, but it was raising an error without changing the sense to feasibility first
    # set_objective_sense(player.X, JuMP.MOI.FEASIBILITY_SENSE)
    # @objective(player.X, JuMP.MOI.MAX_SENSE, obj)

    # set_silent(player.X)
    # optimize!(player.X)

    # return value.(xp)
# end
