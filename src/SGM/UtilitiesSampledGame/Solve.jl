
using .MultiGame
using .NashAlgorithms

function NGame(sg::UtilitiesSampledGame)::MultiGame.NGame
    return create_game(
        length(collect(keys(sg.S_X))),          # n
        [length(xs) for xs in values(sg.S_X)],  # strat
        sg.utilities                            # utilities
    )
end

"Compute a (mixed) nash equilibrium for the sampled game using MultiGame."
function solve_multigame(sampled_game::UtilitiesSampledGame, optimizer_factory)::Profile{DiscreteMixedStrategy}
    game = NGame(sampled_game)

    # TODO: pass optimizer_factory to compute_nash_equilibria. this will likely need some modifications of MultiPlayer
    # The above will be much trickier than I initially thought. MultiGame uses Gurobi,
    # Juniper, and Ipopt extensively, changing solver-specific parameters. Also, it uses
    # Gurobi's PWL features under some settings. I believe we would need to dissect
    # MultiGame quite a bit to make it solver-agnostic.
    # For now, I added Gurobi (and Juniper (and Ipopt)) as a dependency :(
    results_game = compute_nash_equilibria(game)

    ## Extract DiscreteMixedStrategy from results_game
    # should be the probabilities at equilibrium for each strategy in the sampled game
    probs = results_game[3]

    return Profile{DiscreteMixedStrategy}(
        p => DiscreteMixedStrategy(probs[i], sampled_game.S_X[p])
        for (i, p) in enumerate(keys(sampled_game.S_X))
    )
end

"""
SGM subroutine for solving utility-based sampled games.

The current implementation is an interface for the solution methods in `MultiGame`.

# Options
 - `solve_multigame` (default)

# Examples
```julia
IPG.solve_utilities_game = IPG.solve_multigame
```
"""
solve_utilities_game = solve_multigame  # default value
public solve_utilities_game, solve_multigame
