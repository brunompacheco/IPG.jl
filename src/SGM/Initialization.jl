
function initialize_strategies_feasibility(players::Vector{Player}, optimizer_factory=nothing)
    S_X = [Vector{Vector{Float64}}() for _ in players]
    for player in players
        xp_init = start_value.(all_variables(player.Xp))

        if nothing in xp_init
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            xp_init = find_feasible_pure_strategy(player, optimizer_factory)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

function initialize_strategies_player_alone(players::Vector{Player}, optimizer_factory=nothing)
    S_X = [Vector{Vector{Float64}}() for _ in players]

    # mixed profile that simulates players being alone (all others play 0)
    σ_dummy = [DiscreteMixedStrategy([1], [zeros(length(all_variables(player.Xp)))]) for player in players]

    for player in players
        xp_init = start_value.(all_variables(player.Xp))

        if nothing in xp_init
            xp_init = best_response(player, σ_dummy, optimizer_factory)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

initialize_strategies = initialize_strategies_feasibility
