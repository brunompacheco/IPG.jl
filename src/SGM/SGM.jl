
# include("SampledGame/SampledGame.jl")  # TODO: refactor polymatrix out of sampled game. SampledGame now is a simple const
include("PolymatrixGame/PolymatrixGame.jl")
include("PlayerOrder.jl")
include("DeviationReaction.jl")
include("Initialization.jl")


# function SGM(players::Vector{Player}, optimizer_factory; max_iter=100, dev_tol=1e-3, verbose=false)
#     # set `optimizer_factory` the optimizer for each player that doesn't have one yet
#     for player in players
#         # check whether an optimizer has already been set to player
#         if ~has_optimizer(player)
#             set_optimizer(player, optimizer_factory)
#         end
#     end

#     ### Step 1: Initialization
#     # The sampled game (sample of feasible strategies) is built from warm-start values from
#     # the strategy space of each player or, in case there is none, a feasibility problem is
#     # solved

#     # TODO: I should use S_X for the sampled game, rather than the strategies
#     S_X = initialize_strategies(players)
#     sampled_game = SampledGame(players, S_X)
#     verbose && println("Game initialized with strategies: ", S_X)

#     Σ_S = Vector{Profile{DiscreteMixedStrategy}}()  # candidate equilibria
#     payoff_improvements = Vector{Tuple{Integer,Float64}}()
#     for iter in 1:max_iter
#         verbose && println("Iter ", iter)

#         ### Step 2: Solve sampled game
#         # A (mixed) Nash equilibrium is computed for the sampled game. Note that it is a
#         # feasible strategy for the original game, but not necessarily a equilibrium.

#         σ_S = solve(sampled_game, optimizer_factory)
#         push!(Σ_S, σ_S)
#         verbose && println("Sampled game equilibrium found: ", σ_S)

#         ### Step 3: Termination
#         # Find a deviation from `σ` for some player and add it to the sample. If no deviation is
#         # found, `σ` is a equilibrium for the game, so we stop.

#         player_order = get_player_order(players, iter, Σ_S, payoff_improvements)
#         payoff_improvement, p, new_xp = find_deviation(players, σ_S, player_order=player_order, dev_tol=dev_tol)
#         push!(payoff_improvements, (p, payoff_improvement))
#         verbose && println("Deviation found for player ", p, " with payoff improvement ", payoff_improvement)

#         if payoff_improvement < dev_tol
#             break
#         end

#         ### Step 4: Generation of next sampled game
#         # TODO: record deviations
#         add_new_strategy!(sampled_game, players, new_xp, p)
#         verbose && println("New strategy added for player ", p, ": ", new_xp)

#         if iter == max_iter
#             println("Maximum number of iterations reached!")
#         end
#     end

#     # TODO: add verbose option to return all intermediate σ
#     return Σ_S, payoff_improvements
# end