
include("SampledGame/SampledGame.jl")
include("PlayerOrder.jl")
include("DeviationReaction.jl")
include("Initialization.jl")

function SGM(players::Vector{Player}, optimizer_factory=nothing; max_iter=100, dev_tol=1e-3)
    ### Step 1: Initialization
    # The sampled game (sample of feasible strategies) is built from warm-start values from
    # the strategy space of each player or, in case there is none, a feasibility problem is
    # solved

    S_X = initialize_strategies(players, optimizer_factory)
    sampled_game = SampledGame(players, S_X)

    Σ_S = Vector{Vector{DiscreteMixedStrategy}}()  # candidate equilibria
    payoff_improvements = Vector{Tuple{Integer,Float64}}()
    for iter in 1:max_iter
        ### Step 2: Solve sampled game
        # A (mixed) Nash equilibrium is computed for the sampled game. Note that it is a
        # feasible strategy for the original game, but not necessarily a equilibrium.

        σ_S = solve(sampled_game)
        push!(Σ_S, σ_S)

        ### Step 3: Termination
        # Find a deviation from `σ` for some player and add it to the sample. If no deviation is
        # found, `σ` is a equilibrium for the game, so we stop.

        player_order = get_player_order(players, iter, Σ_S, payoff_improvements)
        payoff_improvement, p, new_xp = find_deviation(players, σ_S, optimizer_factory, player_order=player_order, dev_tol=dev_tol)
        push!(payoff_improvements, (p, payoff_improvement))

        if payoff_improvement < dev_tol
            break
        end

        ### Step 4: Generation of next sampled game
        # TODO: record deviations
        add_new_strategy!(sampled_game, players, new_xp, p)

        if iter == max_iter
            println("Maximum number of iterations reached!")
        end
    end

    # TODO: add verbose option to return all intermediate σ
    return Σ_S, payoff_improvements
end