# Integer Programming Games

Implementation of the sampled generation method (SGM) for equilibria computation of integer programming games (IPGs). See:

[*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing equilibria for integer programming games". 2022. European Journal of Operational Research*](https://www.sciencedirect.com/science/article/pii/S0377221722002727)

[*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing Nash equilibria for integer programming games". 2020. arXiv:2012.07082*](https://arxiv.org/abs/2012.07082)

## Example

This example is based on Example 5.3, from Carvalho, Lodi, and Pedroso (2020).
```julia
julia> using IPG

julia> player_1 = Player(QuadraticPayoff(0, [2, 1]), 1);

julia> @variable(player_1.Xp, x1, start=10)
x1

julia> @constraint(player_1.Xp, x1 >= 0)
x1 ≥ 0

julia> player_2 = Player(QuadraticPayoff(0, [1, 2]), 2);

julia> @variable(player_2.Xp, x2, start=10)
x2

julia> @constraint(player_2.Xp, x2 >= 0)
x2 ≥ 0

julia> Σ, payoff_improvements = IPG.SGM([player_1, player_2], Gurobi.Optimizer, max_iter=5);

julia> Σ[end]
2-element Vector{DiscreteMixedStrategy}:
 DiscreteMixedStrategy([1.0], [[1.25]])
 DiscreteMixedStrategy([1.0], [[0.625]])

```


## NormalGames

Note this repository depends on NormalGames.jl to solve normal-form games, which has not been published (yet). Follow the installation instructions at https://github.com/mxmmargarida/Normal-form-games.

