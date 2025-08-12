
empty_S_X(players::Vector{Player}) = Dict{Player, Vector{PureStrategy}}(p => Vector{PureStrategy}() for p in players)

# TODO: refactor strategies to apply to a single player at a time. leave the overwriting of start values outside?

"Solves a feasibility problem for each player individually."
function initialize_strategies_feasibility(players::Vector{Player})
    S_X = empty_S_X(players)
    for player in players
        xp_init = start_value.(all_variables(player))

        if nothing in xp_init
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            xp_init = find_feasible_pure_strategy(player)
        end

        push!(S_X[player], xp_init)
    end

    return S_X
end

"Computes the best response of each player when others play 0."
function initialize_strategies_player_alone(players::Vector{Player})
    S_X = empty_S_X(players)

    # profile that simulates players being alone (all others play 0)
    x_dummy = Profile{PureStrategy}(player => zeros(length(all_variables(player))) for player in players)

    for player in players
        xp_init = start_value.(all_variables(player))

        if nothing in xp_init
            xp_init = best_response(player, others(x_dummy, player))
        end

        push!(S_X[player], xp_init)
    end

    return S_X
end

"""
Define how initial strategies are computed for each player.

In all current options, initialization is only applied to players that do *not* have start
value for *all* variables, i.e., whenever `all(has_start_value.(all_variables(player))) == false`.

# Options
 - `initialize_strategies_feasibility` (default)
 - `initialize_strategies_player_alone`

# Examples
```julia
IPG.initialize_strategies = IPG.initialize_strategies_feasibility
```
"""
initialize_strategies = initialize_strategies_feasibility
public initialize_strategies, initialize_strategies_player_alone, initialize_strategies_feasibility
