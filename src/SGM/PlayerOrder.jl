
const PayoffImprovements = Vector{Tuple{Player, Float64}}
const CandidateEquilibria = Vector{Profile{DiscreteMixedStrategy}}

function get_player_order_fixed(
    players::Vector{Player},
    iter::Integer,
    Σ_S::CandidateEquilibria,
    payoff_improvements::PayoffImprovements
)
    return keys(players)
end

function get_player_order_fixed_ascending(
    players::Vector{Player},
    iter::Integer,
    Σ_S::CandidateEquilibria,
    payoff_improvements::PayoffImprovements
)
    return sort(keys(players))
end

function get_player_order_fixed_descending(
    players::Vector{Player},
    iter::Integer,
    Σ_S::CandidateEquilibria,
    payoff_improvements::PayoffImprovements
)
    return reverse(sort(keys(players)))
end

function get_player_order_random(
    players::Vector{Player},
    iter::Integer,
    Σ_S::CandidateEquilibria,
    payoff_improvements::PayoffImprovements
)
    return shuffle(keys(players))
end

function get_player_order_by_last_deviation(
    players::Vector{Player},
    iter::Integer,
    Σ_S::CandidateEquilibria,
    payoff_improvements::PayoffImprovements
)
    iterations_since_last_deviation = Dict(p => length(payoff_improvements) for p in players)

    for i in 0:length(payoff_improvements)-1
        p, _ = payoff_improvements[end-i]

        if iterations_since_last_deviation[p] > i
            iterations_since_last_deviation[p] = i
        end
    end

    # sort by the number of iterations since the last deviation (decreasing)
    iterations_since_last_deviation = sort(collect(iterations_since_last_deviation), by = x -> -x[2])

    return [i for (i, _) in enumerate(iterations_since_last_deviation)]
end

# TODO: instead of having a player order, just reorder the list of players
get_player_order = get_player_order_by_last_deviation  # default value
