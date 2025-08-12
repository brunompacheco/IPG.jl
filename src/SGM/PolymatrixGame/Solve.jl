using NormalGames

"Convert our polymatrix representation to the one used in NormalGames.jl (integer indices)."
function polymatrix_to_NG_std(polymatrix::Polymatrix, players::Vector{Player})::Dict{Tuple{Int, Int}, Matrix{Float64}}
    return Dict(
        (i_p, i_k) => polymatrix[p,k]
        for (i_p, p) in enumerate(players)
        for (i_k, k) in enumerate(players)
    )
end

function discrete_profile_from_NE(NE_mixed::Vector{Vector{Float64}}, S_X::Dict{Player, Vector{PureStrategy}})::Profile{DiscreteMixedStrategy}
    players = collect(keys(S_X))
    NE_probs = Dict(p => probs for (p, probs) in zip(players, NE_mixed))

    # build mixed strategy profile from the NE of the normal game
    return Profile{DiscreteMixedStrategy}(
        p => DiscreteMixedStrategy(NE_probs[p], S_X[p])
        for p in players
    )
end

"Compute a (mixed) nash equilibrium for the sampled game using PNS."
function solve_PNS(sampled_game::PolymatrixSampledGame, optimizer_factory)::Profile{DiscreteMixedStrategy}
    players = collect(keys(sampled_game.S_X))

    normal_game = NormalGames.NormalGame(
        length(players),
        [length(sampled_game.S_X[p]) for p in players],
        polymatrix_to_NG_std(sampled_game.polymatrix, players)
    )

    _, _, NE_mixed = NormalGames.NashEquilibriaPNS(normal_game, optimizer_factory, false, false, false)

    # each element in NE_mixed is a mixed NE, represented as a vector of probabilities in 
    # the same shape as S_X
    NE_mixed = NE_mixed[1]  # take the first NE

    return discrete_profile_from_NE(NE_mixed, sampled_game.S_X)
end

"Compute a (mixed) nash equilibrium for the sampled game using Formulation 1 (Big-M) of Sandholm et al. (2005)."
function solve_Sandholm1(sampled_game::PolymatrixSampledGame, optimizer_factory)::Profile{DiscreteMixedStrategy}
    players = collect(keys(sampled_game.S_X))
    polymatrix = polymatrix_to_NG_std(sampled_game.polymatrix, players)

    # the method doesn't support polymatrices with negative entries, so a quick
    # preprocessing is performed
    offset = 0
    for k in keys(polymatrix)
        offset = min(offset, minimum(polymatrix[k]))
    end
    for k in keys(polymatrix)
        polymatrix[k] = polymatrix[k] .- offset
    end
    normal_game = NormalGames.NormalGame(
        length(players),
        [length(sampled_game.S_X[p]) for p in players],
        polymatrix
    )

    _, _, NE_mixed, _, _, _, _, _ = NormalGames.NashEquilibria2(normal_game, optimizer_factory)

    return discrete_profile_from_NE(NE_mixed, sampled_game.S_X)
end

"""
Interface for the solution methods in `NormalGames.jl` for polymatrix sampled games.

# Options
 - `solve_PNS` (default)
 - `solve_Sandholm1`

# Examples
```julia
IPG.solve = IPG.solve_PNS
```
"""
solve = solve_PNS  # default value
public solve, solve_PNS, solve_Sandholm1
