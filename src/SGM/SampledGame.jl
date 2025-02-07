
"Compute polymatrix for normal form game from sample of strategies."
function get_polymatrix(players::Vector{Player}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    polymatrix = Dict{Tuple{Integer, Integer}, Matrix{Float64}}()

    # compute utility of each player `p` using strategy `i_p` against player `k` using strategy `i_k`
    for player_p in players
        for player_k in players
            p = player_p.p
            k = player_k.p

            polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))

            if k != p
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

"Wrapper for NormalGames.NormalGame that includes the sample of strategies."
mutable struct SampledGame
    S_X::Vector{Vector{Vector{Float64}}}  # sample of strategies (finite subset of the strategy space X)
    normal_game::NormalGames.NormalGame
end
function SampledGame(players::Vector{Player}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    payoff_polymatrix = get_polymatrix(players, S_X)
    normal_game = NormalGames.NormalGame(length(players), length.(S_X), payoff_polymatrix)

    return SampledGame(S_X, normal_game)
end

"[SearchNE] Compute a (mixed) nash equilibrium for the sampled game."
function solve(sampled_game::SampledGame)::Vector{DiscreteMixedStrategy}
    _, _, NE_mixed = NormalGames.NashEquilibriaPNS(sampled_game.normal_game, false, false, false)
    # each element in NE_mixed is a mixed NE, represented as a vector of probabilities in 
    # the same shape as S_X

    NE_mixed = NE_mixed[1]  # take the first NE

    # build mixed strategy profile from the NE of the normal game
    σ = [IPG.DiscreteMixedStrategy(probs_p, S_Xp) for (probs_p, S_Xp) in zip(NE_mixed, sampled_game.S_X)]

    return σ
end

function add_new_strategy!(sampled_game::SampledGame, players::Vector{Player}, new_xp::Vector{<:Real}, p::Integer)
    # first part is easy, just add the new strategy to the set
    push!(sampled_game.S_X[p], new_xp)

    n = sampled_game.normal_game.n
    strat = sampled_game.normal_game.strat
    polymatrix = sampled_game.normal_game.polymatrix

    strat[p] += 1

    # now we need to update the normal game (polymatrix)
    for (p1, p2) in keys(polymatrix)
        if p1 == p2 == p
            # add new row to polymatrix to store the utilities wrt the new strategy
            polymatrix[p,p] = zeros(strat[p], strat[p])
        elseif (p1 != p) & (p2 == p)
            # add new column to polymatrix to store the utilities wrt the new strategy
            polymatrix[p1,p] = hcat(polymatrix[p1,p], zeros(strat[p1], 1))

            for i in 1:strat[p1]
                # compute utility of player `p1` using strategy `i` against the new strategy of player `p`
                polymatrix[p1,p][i,end] = (
                    IPG.bilateral_payoff(players[p1], sampled_game.S_X[p1][i], players[p], new_xp)
                    + IPG.bilateral_payoff(players[p1], sampled_game.S_X[p1][i], players[p1], sampled_game.S_X[p1][i])
                )
            end
        elseif (p1 == p) & (p2 != p)
            # add new row to polymatrix to store the utilities wrt the new strategy
            polymatrix[p,p2] = vcat(polymatrix[p,p2], zeros(1, strat[p2]))

            for i in 1:strat[p2]
                # compute utility of player `p1` using strategy `i` against the new strategy of player `p`
                polymatrix[p,p2][end,i] = (
                    IPG.bilateral_payoff(players[p], new_xp, players[p2], sampled_game.S_X[p2][i])
                    + IPG.bilateral_payoff(players[p], new_xp, players[p], new_xp)
                )
            end
        end
    end

    sampled_game.normal_game = NormalGames.NormalGame(n, strat, polymatrix)
end
