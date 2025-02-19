
"Find a deviation from mixed profile `σ`."
function find_deviation_best_response(players::Vector{Player}, σ::Vector{DiscreteMixedStrategy}; player_order=nothing, dev_tol=1e-3)::Tuple{Float64,Int64,Union{Nothing,Vector{Float64}}}
    if isnothing(player_order)
        player_order = 1:length(players)
    end

    for p in player_order
        player = players[p]
        new_x_p = best_response(player, σ)

        new_σ = copy(σ)
        new_σ[p] = DiscreteMixedStrategy([1], [new_x_p])
        payoff_improvement = payoff(player, new_σ) - payoff(player, σ)
        if payoff_improvement > dev_tol
            return payoff_improvement, p, new_x_p
        end
    end

    return 0.0, -1, nothing
end

find_deviation = find_deviation_best_response  # default value
