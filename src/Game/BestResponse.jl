
function payoff(player::Player{<:AbstractJuMPScalar}, x_player::Vector{<:Any}, x_others::Profile{<:Vector{<:Real}})
    variable_assignments = Dict{VariableRef, Any}()
    for (p, xp) in x_others
        for (v, val) in zip(all_variables(p.Xp), xp)
            variable_assignments[v] = val
        end
    end
    for (v, val) in zip(all_variables(player.Xp), x_player)
        variable_assignments[v] = val
    end

    return value(v -> get(variable_assignments, v, 0), player.Πp)
end

"Compute `player`'s best response to the pure strategy profile `x_others`."
function best_response(player::Player{<:AbstractPayoff}, x_others::Profile{<:Vector{<:Real}})
    dummy_mixed_profile = Profile{DiscreteMixedStrategy}(
        p => DiscreteMixedStrategy(xp) for (p, xp) in x_others
    )
    return best_response(player, dummy_mixed_profile)
end

"Compute `player`'s best response to the mixed strategy profile `σ_others`."
function best_response(player::Player{<:AbstractJuMPScalar}, σ_others::Profile{DiscreteMixedStrategy})
    x_player = all_variables(player.Xp)

    obj = expected_value(x_others -> payoff(player, x_player, x_others), σ_others)

    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.Xp, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.Xp, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(x_player)
end
function best_response(player::Player{<:AbstractBilateralPayoff}, σ::Vector{DiscreteMixedStrategy})
    error("best_response for player with bilateral payoff not implemented yet") # TODO

    xp = all_variables(player.Xp)

    # TODO: No idea why this doesn't work
    # @objective(model, Max, sum([IPG.bilateral_payoff(Πp, p, xp, k, σ[k]) for k in 1:m]))

    obj = AffExpr()
    for k in eachindex(σ)
        if k == player.p
            obj += IPG.bilateral_payoff(player.Πp, xp)
        else
            obj += IPG.bilateral_payoff(player.Πp, xp, σ[k], k)
        end
    end
    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.Xp, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.Xp, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(xp)
end
