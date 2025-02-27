using NormalGames

"Compute a (mixed) nash equilibrium for the sampled game using PNS."
function solve_PNS(sampled_game::SampledGame, optimizer_factory)::Vector{DiscreteMixedStrategy}
    normal_game = NormalGames.NormalGame(
        length(sampled_game.S_X),
        length.(sampled_game.S_X),
        sampled_game.polymatrix
    )

    _, _, NE_mixed = NormalGames.NashEquilibriaPNS(normal_game, optimizer_factory, false, false, false)

    # each element in NE_mixed is a mixed NE, represented as a vector of probabilities in 
    # the same shape as S_X
    NE_mixed = NE_mixed[1]  # take the first NE

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(probs_p, S_Xp) for (probs_p, S_Xp) in zip(NE_mixed, sampled_game.S_X)]

    return σ
end

"Compute a (mixed) nash equilibrium for the sampled game using Formulation 1 (Big-M) of Sandholm et al. (2005)."
function solve_Sandholm1(sampled_game::SampledGame, optimizer_factory)::Vector{DiscreteMixedStrategy}
    # the method doesn't support polymatrices with negative entries, so a quick
    # preprocessing is performed
    polymatrix = copy(sampled_game.polymatrix)
    offset = 0
    for k in keys(polymatrix)
        offset = min(offset, minimum(polymatrix[k]))
    end
    for k in keys(polymatrix)
        polymatrix[k] = polymatrix[k] .- offset
    end
    normal_game = NormalGames.NormalGame(
        length(sampled_game.S_X),
        length.(sampled_game.S_X),
        polymatrix
    )

    _, _, NE_mixed, _, _, _, _, _ = NormalGames.NashEquilibria2(normal_game, optimizer_factory)

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(probs_p, S_Xp) for (probs_p, S_Xp) in zip(NE_mixed, sampled_game.S_X)]

    return σ
end

solve = solve_PNS  # default value