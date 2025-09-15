
abstract type AbstractStrategyInit end

empty_S_X(players::Vector{Player}) = Dict{Player, Vector{PureStrategy}}(p => Vector{PureStrategy}() for p in players)

# TODO: refactor strategies to apply to a single player at a time. leave the overwriting of start values outside?

"Solves a feasibility problem for each player individually."
struct FeasibilityStrategyInit <: AbstractStrategyInit end
export FeasibilityStrategyInit

function initialize_strategies(::FeasibilityStrategyInit, players::Vector{Player})
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
struct PlayerAloneStrategyInit <: AbstractStrategyInit end
export PlayerAloneStrategyInit

function initialize_strategies(::PlayerAloneStrategyInit, players::Vector{Player})
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

""" Default strategy initialization method.

Options:
- `FeasibilityStrategyInit()` (default)
- `PlayerAloneStrategyInit()`

"""
DEFAULT_STRATEGY_INITIALIZER = FeasibilityStrategyInit()
public DEFAULT_STRATEGY_INITIALIZER

"""
SGM subroutine that computes initial strategies for each player.

In all current options, initialization is only applied to players that do *not* have start
value for *all* variables, i.e., whenever `all(has_start_value.(all_variables(player))) == false`.


# Options
 - `FeasibilityStrategyInit()` (default)
 - `PlayerAloneStrategyInit()`


# Examples
```julia
# Use a specific initializer for one call
S_X = initialize_strategies(PlayerAloneStrategyInit(), players)

# Change the default initializer globally
IPG.DEFAULT_STRATEGY_INITIALIZER = PlayerAloneStrategyInit()
S_X = initialize_strategies(players)  # now uses PlayerAloneStrategyInit by default
```
"""
initialize_strategies(players::Vector{Player}) = initialize_strategies(DEFAULT_STRATEGY_INITIALIZER, players)
initialize_strategies(init::AbstractStrategyInit, players::Vector{Player}) = initialize_strategies(init, players)

public initialize_strategies
