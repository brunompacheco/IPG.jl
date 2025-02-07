
function get_player_order_fixed_ascending(players::Vector{Player}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Real})
    return keys(players)
end

function get_player_order_fixed_descending(players::Vector{Player}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Real})
    return reverse(keys(players))
end

get_player_order = get_player_order_fixed_ascending  # default value
