
"Find a deviation from mixed profile `σ` by computing best responses following `player_order`."
function find_deviation_best_response(players::Vector{Player}, σ::Profile{DiscreteMixedStrategy}; player_order=nothing, dev_tol=1e-3)
    player_order = isnothing(player_order) ? eachindex(players) : player_order

    for p in player_order
        player = players[p]
        σ_others = others(σ, player)

        new_x_p = best_response(player, σ_others)

        payoff_improvement = payoff(player, new_x_p, σ_others) - payoff(player, σ)
        if payoff_improvement > dev_tol
            return payoff_improvement, player, new_x_p
        end
    end

    return 0.0, nothing, nothing
end

"""Find a deviation from a candidate equilibria `σ` (mixed profile).

Current options:
 - `find_deviation_best_response` (default)

Example usage:
```
> IPG.find_deviation = IPG.find_deviation_best_response
```
"""
find_deviation = find_deviation_best_response  # default value
public find_deviation, find_deviation_best_response
