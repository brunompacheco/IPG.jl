
"Compute a (mixed) nash equilibrium for the sampled game using PNS."
function solve_PNS(sampled_game::SampledGame)::Vector{DiscreteMixedStrategy}
    _, _, NE_mixed = NormalGames.NashEquilibriaPNS(sampled_game.normal_game, false, false, false)

    # each element in NE_mixed is a mixed NE, represented as a vector of probabilities in 
    # the same shape as S_X
    println("Normal game solution: ", NE_mixed)

    NE_mixed = NE_mixed[1]  # take the first NE

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(probs_p, S_Xp) for (probs_p, S_Xp) in zip(NE_mixed, sampled_game.S_X)]

    return σ
end

"Compute a (mixed) nash equilibrium for the sampled game using Formulation 1 (Big-M) of Sandholm et al. (2005)."
function solve_Sandholm1(sampled_game::SampledGame)::Vector{DiscreteMixedStrategy}
    # the method doesn't support polymatrices with negative entries, so a quick
    # preprocessing is performed
    polymatrix = copy(sampled_game.normal_game.polymatrix)
    offset = 0
    for k in keys(polymatrix)
        offset = min(offset, minimum(polymatrix[k]))
    end
    for k in keys(polymatrix)
        polymatrix[k] = polymatrix[k] .- offset
    end
    normal_game = NormalGames.NormalGame(sampled_game.normal_game.n, sampled_game.normal_game.strat, polymatrix)

    _, _, NE_mixed, _, _, _, _, _ = NormalGames.NashEquilibria2(normal_game)

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(probs_p, S_Xp) for (probs_p, S_Xp) in zip(NE_mixed, sampled_game.S_X)]

    return σ
end

solve = solve_PNS  # default value