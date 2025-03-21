
function initialize_strategies_feasibility(players::Vector{<:AbstractPlayer})
    S_X = [Vector{PureStrategy}() for _ in players]
    for player in players
        xp_init = [start_value.(v) for v in variables(player)]

        if nothing in xp_init
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            xp_init = find_feasible_pure_strategy(player)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

function initialize_strategies_player_alone(players::Vector{<:AbstractPlayer})
    S_X = [Vector{Vector{Any}}() for _ in players]

    # mixed profile that simulates players being alone (all others play 0)
    σ_dummy = [DiscreteMixedStrategy([1], [(v isa VariableRef ? [0] : zeros(size(v))) for v in variables(player)]) for player in players]

    for player in players
        xp_init = [start_value.(v) for v in variables(player)]

        if nothing in xp_init
            xp_init = best_response(player, σ_dummy)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

initialize_strategies = initialize_strategies_feasibility
