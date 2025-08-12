
# include("SampledGame/SampledGame.jl")  # TODO: refactor polymatrix out of sampled game. SampledGame now is a simple const
include("PolymatrixGame/PolymatrixGame.jl")
include("PlayerOrder.jl")
include("DeviationReaction.jl")
include("Initialization.jl")


"""
    SGM(players, optimizer_factory, max_iter=100, dev_tol=1e-3, verbose=false)

Implementation of the _sampled generation method_ (SGM) for simultaneous games.

See
- [*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing equilibria for integer programming games". 2022. European Journal of Operational Research*](https://www.sciencedirect.com/science/article/pii/S0377221722002727)
- [*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing Nash equilibria for integer programming games". 2020. arXiv:2012.07082*](https://arxiv.org/abs/2012.07082)
for further details on the algorithm.

# Arguments
 - `players::Vector{Player}`: players in the simultaneous game.
 - `optimizer_factory`: JuMP-suitable optimizer used to solve the sampled games. Also used in case the player's optimizer is not set.
 - `max_iter::UInt`: maximum number of iterations.
 - `dev_tol::Float64`: tolerance for deviation detection. If a deviation is found with a payoff improvement less than this value, the algorithm stops.
 - `verbose::Bool`: whether to print information about the algorithm progress.

# Examples
See the [examples folder](IPG.jl/examples/).
"""
function SGM(players::Vector{Player}, optimizer_factory;
             max_iter::Integer=100, dev_tol::Float64=1e-3, verbose::Bool=false)
    # set `optimizer_factory` the optimizer for each player that doesn't have one yet
    for player in players
        # check whether an optimizer has already been set to player
        if ~has_optimizer(player)
            set_optimizer(player, optimizer_factory)
        end
    end

    ### Step 1: Initialization
    # The sampled game (sample of feasible strategies) is built from warm-start values from
    # the strategy space of each player or, in case there is none, a feasibility problem is
    # solved

    S_X = initialize_strategies(players)
    sampled_game = PolymatrixSampledGame(players, S_X)
    verbose && println("Game initialized with strategies: ", S_X)

    Σ_S = Vector{Profile{DiscreteMixedStrategy}}()  # candidate equilibria
    payoff_improvements = Vector{Tuple{Player,Float64}}()
    for iter in 1:max_iter
        verbose && println("Iter ", iter)

        ### Step 2: Solve sampled game
        # A (mixed) Nash equilibrium is computed for the sampled game. Note that it is a
        # feasible strategy for the original game, but not necessarily a equilibrium.

        σ_S = solve(sampled_game, optimizer_factory)
        push!(Σ_S, σ_S)
        verbose && println("Sampled game equilibrium found: ", σ_S)

        ### Step 3: Termination
        # Find a deviation from `σ` for some player and add it to the sample. If no deviation is
        # found, `σ` is a equilibrium for the game, so we stop.

        player_order = get_player_order(players, iter, Σ_S, payoff_improvements)
        payoff_improvement, p, new_xp = find_deviation(players, σ_S, player_order=player_order, dev_tol=dev_tol)

        if payoff_improvement < dev_tol
            verbose && println("No deviation found (within `dev_tol`). It's an equilibrium!")
            break
        else
            push!(payoff_improvements, (p, payoff_improvement))
            verbose && println("Deviation found for player ", p, " with payoff improvement ", payoff_improvement)
        end

        ### Step 4: Generation of next sampled game
        # TODO: record deviations
        add_new_strategy!(sampled_game, p, new_xp)
        verbose && println("New strategy added for player ", p, ": ", new_xp)

        if iter == max_iter
            println("Maximum number of iterations reached!")
        end
    end

    # TODO: add verbose option to return all intermediate σ
    return Σ_S, payoff_improvements
end
export SGM
