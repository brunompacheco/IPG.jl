
"Compute polymatrix for normal form game from sample of strategies."
function get_polymatrix(players::Vector{Player}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    polymatrix = Dict{Tuple{Integer, Integer}, Matrix{Float64}}()

    # compute utility of each player `p` using strategy `i_p` against player `k` using strategy `i_k`
    for player_p in players
        for player_k in players
            p = player_p.p
            k = player_k.p

            polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))

            if k != p  # fills only the diagonal
                for i_p in 1:length(S_X[p])
                    for i_k in 1:length(S_X[k])
                        polymatrix[p,k][i_p,i_k] = (
                            IPG.bilateral_payoff(players[p], S_X[p][i_p], players[k], S_X[k][i_k])
                            + IPG.bilateral_payoff(players[p], S_X[p][i_p], players[p], S_X[p][i_p])
                        )
                    end
                end
            end
        end
    end

    return polymatrix
end

"[SearchNE] Compute a (mixed) nash equilibrium for the sampled game."
function solve_sampled_game(players::Vector{Player}, S_X::Vector{<:Vector{<:Vector{<:Real}}})::Vector{DiscreteMixedStrategy}
    # build normal form game from sample of strategies
    G = NormalGames.NormalGame(length(players), length.(S_X), get_polymatrix(players, S_X))
    # TODO: the polymatrix doesnt need to be recomputed every time, we can just update it
    # This will probably require a new struct (maybe `SampledGame`) that holds the samples
    # and the normal game

    _, _, NE_mixed = NormalGames.NashEquilibriaPNS(G,false,false,false)
    # each element in NE_mixed is a mixed NE, represented as a vector of probabilities in 
    # the same shape as S_X

    NE_mixed = NE_mixed[1]  # take the first NE

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(NE_mixed[player.p], S_X[player.p]) for player in players]

    return σ
end

"[DeviationReaction] Compute `player`'s best response to the mixed strategy profile `σp`."
function best_response(player::Player, σ::Vector{DiscreteMixedStrategy}, optimizer_factory=nothing)
    if ~isnothing(optimizer_factory)
        set_optimizer(player.Xp, optimizer_factory)
    end

    xp = all_variables(player.Xp)

    # TODO: No idea why this doesn't work
    # @objective(model, Max, sum([IPG.bilateral_payoff(Πp, p, xp, k, σ[k]) for k in 1:m]))

    obj = QuadExpr()
    for k in 1:length(σ)
        obj += IPG.bilateral_payoff(player.Πp, player.p, xp, k, σ[k])
    end
    @objective(player.Xp, Max, obj)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(xp)
end

"Find a deviation from mixed profile `σ`."
function find_deviation(players::Vector{Player}, σ::Vector{DiscreteMixedStrategy}, optimizer_factory=nothing; player_order=nothing, dev_tol=1e-3)::Tuple{Float64,Int64,Union{Nothing,Vector{Float64}}}
    if isnothing(player_order)
        player_order = 1:length(players)
    end

    for p in player_order
        player = players[p]
        new_x_p = best_response(player, σ, optimizer_factory)

        new_σ = copy(σ)
        new_σ[p] = DiscreteMixedStrategy([1], [new_x_p])
        payoff_improvement = payoff(player, new_σ) - payoff(player, σ)
        if payoff_improvement > dev_tol
            return payoff_improvement, p, new_x_p
        end
    end

    return 0.0, -1, nothing
end

function SGM(players::Vector{Player}, optimizer_factory=nothing; max_iter=100, dev_tol=1e-3)
    ### Step 1: Initialization
    # The sampled game (sample of feasible strategies) is built from warm-start values from
    # the strategy space of each player or, in case there is none, a feasibility problem is
    # solved

    S_X = [Vector{Vector{Float64}}() for _ in players]
    for player in players
        initial_strat = start_value.(all_variables(player.Xp))

        if nothing in initial_strat
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            initial_strat = find_feasible_pure_strategy(player, optimizer_factory)
        end

        push!(S_X[player.p], initial_strat)
    end

    Σ_S = Vector{Vector{DiscreteMixedStrategy}}()
    payoff_improvements = Vector{Float64}()
    for _ in 1:max_iter
        ### Step 2: Solve sampled game
        # A (mixed) Nash equilibrium is computed for the sampled game. Note that it is a
        # feasible strategy for the original game, but not necessarily a equilibrium.
        σ_S = solve_sampled_game(players, S_X)
        push!(Σ_S, σ_S)

        ### Step 3: Termination
        # Find a deviation from `σ` for some player and add it to the sample. If no deviation is
        # found, `σ` is a equilibrium for the game, so we stop.
        payoff_improvement, p, new_xp = find_deviation(players, σ_S, optimizer_factory, dev_tol=dev_tol)
        push!(payoff_improvements, payoff_improvement)

        if payoff_improvement < dev_tol
            break
        end

        ### Step 4: Generation of next sampled game
        # IMPORTANT TODO: record deviations!
        push!(S_X[p], new_xp)
    end
    println("Maximum number of iterations reached!")

    # TODO: add verbose option to return all intermediate σ
    return Σ_S, payoff_improvements
end