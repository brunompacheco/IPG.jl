using SCIP, IPG

# ==== Player Definition ====
# Strategy spaces are defined through JuMP models
player1 = Player()
@variable(player1.X, x1, start=10.0)
@constraint(player1.X, x1 >= 0)

player2 = Player()
@variable(player2.X, x2, start=10.0)
@constraint(player2.X, x2 >= 0)

# Payoffs are defined using JuMP expressions
set_payoff!(player1, -x1^2 + x1*x2)
set_payoff!(player2, -x2^2 + x1*x2)

# ==== SGM Algorithm Subroutines ====
# The following are the default options for the SGM algorithm, so they can be omitted.
# See docstrings of each parameter for further details
IPG.initialize_strategies = IPG.initialize_strategies_feasibility
IPG.solve = IPG.solve_PNS
IPG.find_deviation = IPG.find_deviation_best_response

# It is also possible to pass custom subroutines. The following is a re-implementation of
# `IPG.get_player_order_random` for demonstration.
using Random
function get_player_random_order(players::Vector{Player}, iter::Integer, Σ_S::Vector{Profile{DiscreteMixedStrategy}}, payoff_improvements::Vector{Tuple{Player,Float64}})
    return shuffle(keys(players))
end
IPG.get_player_order = get_player_random_order

# ==== SGM ====
players = [player1, player2]
Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=100, verbose=true);
