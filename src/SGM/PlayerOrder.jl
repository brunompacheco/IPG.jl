
function get_player_order_fixed(players::Vector{<:AbstractPlayer}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Tuple{<:Integer,<:Real}})
    return keys(players)
end

function get_player_order_fixed_ascending(players::Vector{<:AbstractPlayer}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Tuple{<:Integer,<:Real}})
    return sort(keys(players))
end

function get_player_order_fixed_descending(players::Vector{<:AbstractPlayer}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Tuple{<:Integer,<:Real}})
    return reverse(sort(keys(players)))
end

function get_player_order_random(players::Vector{<:AbstractPlayer}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Tuple{<:Integer,<:Real}})
    return shuffle(keys(players))
end

function get_player_order_by_last_deviation(players::Vector{<:AbstractPlayer}, iter::Integer, Σ_S::Vector{Vector{DiscreteMixedStrategy}}, payoff_improvements::Vector{<:Tuple{<:Integer,<:Real}})
    iterations_since_last_deviation = Dict(player.p => length(payoff_improvements) for player in players)

    for i in 0:length(payoff_improvements)-1
        p, _ = payoff_improvements[end-i]

        if iterations_since_last_deviation[p] > i
            iterations_since_last_deviation[p] = i
        end
    end

    # sort by the number of iterations since the last deviation (decreasing)
    iterations_since_last_deviation = sort(collect(iterations_since_last_deviation), by=x->-x[2])

    return [p for (p, _) in iterations_since_last_deviation]
end

get_player_order = get_player_order_by_last_deviation  # default value
