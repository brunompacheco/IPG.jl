[![Tests badge](https://github.com/brunompacheco/IPG.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/brunompacheco/IPG.jl/actions/workflows/tests.yml)

# Integer Programming Games

Implementation of the sampled generation method (SGM) for equilibria computation of integer programming games (IPGs). See:

[*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing equilibria for integer programming games". 2022. European Journal of Operational Research*](https://www.sciencedirect.com/science/article/pii/S0377221722002727)

[*M. Carvalho, A. Lodi, J. P. Pedroso, "Computing Nash equilibria for integer programming games". 2020. arXiv:2012.07082*](https://arxiv.org/abs/2012.07082)

## Game definition

A game is any list of players. To define a player, you must define the strategy space and the payoff function. The strategy space is handled through JuMP models, while payoff functions are handled by our structures (see Bilateral payoff games below for examples). Additionally, we need to keep track of references for the decision variables and the index of each player in the game.

The following illustratest the necessary steps to define a player:
```julia
using IPG
Xp = Model()

@variable(Xp, y[1:5], Int)
@variable(Xp, z >= 0)

@constraint(Xp, ...)  # add your constraints

Πp = QuadraticPayoff(...)  # payoff definition

player = Player(Xp, [z], Πp, 1)  # add first player
```
Note that not all variables have to be provided to the Player definition, only those necessary to compute the payoff. <!-- TODO: See payoff section -->

## Payoff Functions

Properly defining the payoff functions are key for the workings of this implementation. Currently, two types of payoff functions are implemented: quadratic and blackbox, which are illustrated in the examples bellow. The former follows the notation of the original paper and allows for any game definition (in terms of constraints) that your solver of choice supports. The latter only works for two-player games. See `src/Game/Payoff` for more details.

## Examples

### Bilateral payoff games

The algorithm is particularly designed for games in which the player have bilaterally-separable payoff functions. This is due to the standard solution method for solving sampled games, which relies on normal form games formulated through the polymatrix. An example of such is the [`QuadraticPayoff`](src/Game/Payoff/BilateralPayoff.jl#L27) structure, which implements the payoff function structure used in the original paper's experiments.

This example is based on Example 5.3, from Carvalho, Lodi, and Pedroso (2020).
```julia
julia> using IPG

julia> X1 = Model();

julia> @variable(X1, x1, start=10)
x1

julia> @constraint(X1, x1 >= 0)
x1 ≥ 0

julia> player_1 = Player(X1, [x1], QuadraticPayoff(0, [2, 1], 1), 1);

julia> X2 = Model();

julia> @variable(X2, x2, start=10)
x2

julia> @constraint(X2, x2 >= 0)
x2 ≥ 0

julia> player_2 = Player(X2, [x2], QuadraticPayoff(0, [1, 2], 2), 2);

julia> Σ, payoff_improvements = IPG.SGM([player_1, player_2], SCIP.Optimizer, max_iter=5);

julia> Σ[end]
2-element Vector{DiscreteMixedStrategy}:
 DiscreteMixedStrategy([1.0], [[0.625]])
 DiscreteMixedStrategy([1.0], [[1.25]])

```
Further details in [`example-5.3.ipynb`](notebooks/example-5.3.ipynb).

### Two-player games

A particular case of bilateral payoff functions that can be handled more generally are two-player games. In this case, because any payoff function is bilateral, we handle the more general [`BlackBoxPayoff`](src/Game/Payoff/Payoff.jl#29) through the SGM algorithm.

Example 5.3 implemented using the BlackBoxPayoff structure.
```julia
julia> using IPG, SCIP

julia> player_payoff(xp, x_others) = -(xp[1] * xp[1]) + xp[1] * prod(x_others[:][1])
player_payoff (generic function with 1 method)

julia> X1 = Model(); @variable(X1, x1, start=10); @constraint(X1, x1 >= 0);

julia> X2 = Model(); @variable(X2, x2, start=10); @constraint(X2, x2 >= 0);

julia> players = [
           Player(X1, [x1], BlackBoxPayoff(player_payoff), 1),
           Player(X2, [x2], BlackBoxPayoff(player_payoff), 2)
       ];

julia> Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5);

julia> Σ[end]
2-element Vector{DiscreteMixedStrategy}:
 DiscreteMixedStrategy([1.0], [[0.625]])
 DiscreteMixedStrategy([1.0], [[1.25]])

```

## Customization

Many components of the algorithm can be modified, as is already discussed in the original work (Table 1 and Section 6.2, Carvalho, Lodi, and Pedroso, 2020). To choose between different options, you have only to assign different implementations to the baseline pointer. Note that those different implementations can be custom, local functions as well.

A practical example is shown in notebook [`example-5.3.ipynb`](./example-5.3.ipynb), at section _Customization_. Below, we detail the customizable parts and the available options.

### Initialization

A initial strategy for each player is necessary to start the SGM algorithm. If _all_ player's variables are assigned start values through JuMP's interface, that will be used as the initial strategy.

Otherwise, the default procedure is to solve the feasibility problem for each player (see [`initialize_strategies_feasibility`](src/SGM/Initialization.jl#L2)).
We have also implemented the initialization procedure that solves the best strategy for each player when she is alone (other strategies set to zero).
To change to this alternative approach, you just need to add `IPG.initialize_strategies = IPG.initialize_strategies_player_alone` before calling the SGM method.

### Sampled game solution

By default, we solve the sampled games using NormalGames.jl, which has not been published (yet). Ask for permission and follow the installation instructions at https://github.com/mxmmargarida/Normal-form-games. Any nash equilibria method can be used. By default, we have wrapped NormalGames' implentation of Porter, Nudelman and Shoham's method in [`solve_PNS`](src/SGM/SampledGame/SearchNE.jl#L3). We also provide a wrapper for the Big-M formulation of Sandholm et al. (2005) in [`solve_Sandholm1`](src/SGM/SampledGame/SearchNE.jl#L19). To choose the latter instead of the former, just assign it to `IPG.solve`.

### Player order

SGM iterates over the players until a deviation from the current candidate equilibrium is found. The order in which this iteration happens can deeply impact the performance _and_ the equilibrium found. The default approach is to order the players in a descending order of the number of iterations since that player found a deviation ([`get_player_order_by_last_deviation`](src/SGM/PlayerOrder.jl#L18)). Other simples methods are also implemented. To choose between them or to simply use your own, assign a different method to `IPG.get_player_order`.

### Deviation

A deviation from the candidate equilibrium is found by computing a player's best response. This is implemented in [`find_deviation_best_response`](src/SGM/DeviationReaction.jl#L3), and any solver can be used to solve these optimization problems, as is going to be discussed below. To use a custom method, assign it to `IPG.find_deviation`.

### MIP Solver

MIP solvers are used (in the default methods) for the initialization of strategies and for computing a deviation (best response, by default) from the candidate equilibrium. Any JuMP-supported MIP solver that can handle the players' problem can be used in SGM. For example, it may be that your players have quadratic constraints, so you will need a MIQCP solver. Your call will look like `IGP.SGM(my_players, MySolver.Optimizer)`.

The algorithm also supports using a different solver for each player. You just need to initialize each player's strategy space with that optimizer factory or use JuMP's method, e.g., `set_optimizer(player, MySolver.Optimizer)`. It is important to note that the `optimizer_factory` that is provided to SGM will _not_ overwrite any player's optimizer. It will be set as the player's optimizer only if no optimizer has been set.
