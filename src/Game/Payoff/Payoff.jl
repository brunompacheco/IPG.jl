using JuMP

"""
A payoff function must implement some form of payoff computation for each player in a game.

A custom implementation can have any fields, but it must implement the `payoff` function.
The function must have the following signature:
```julia
payoff(Πp::MyPayoff, xp::Vector{<:Any}, x_others::Vector{<:Vector{<:Any}})
,```
where `xp` is the strategy of the payoff's player and `x_others` is the strategy profile of
the other players.

Note that the `xp` argument can also be a vector of JuMP variable references. This is
because the payoff functions are also used to build JuMP expressions, e.g., in
`best_response` (see `src/Game/Player.jl`).
"""
abstract type AbstractPayoff end

"""
A wrapper for a black-box payoff function `f`.

The function `f` must have a signature
```julia
f(xp::PureStrategy, x_others::Vector{<:PureStrategy})
,```
similar to the payoff function signature. Note that PureStrategy is a type alias for
`Vector{<:Any}`. The function, however, should work both for the numerical types of the
player's variables AND for JuMP variable references.
"""
struct BlackBoxPayoff <: AbstractPayoff
    "The arguments must be strategy (self) and a strategy profile (others)."
    f::Function
end
# TODO: when building the payoff for a two-player game, it feels a bit weird to have to
# define a function over x_otherS. Maybe we could have a special definition for two-player
# payoffs in which the payoff function might be defined over the two strategies directly?

"Compute the payoff of playing strategy `xp` given the other players play `x_others`."
function payoff(Πp::BlackBoxPayoff, xp::PureStrategy, x_others::Vector{<:PureStrategy})
    return Πp.f(xp, x_others)
end

include("BilateralPayoff.jl")

# Utils

# TODO: check if declaring return type ::Real is useful.
"Compute the payoff of playing `xp` given that the other players play `x_others`."
function payoff(Πp::AbstractPayoff, xp::PureStrategy, σ_others::Vector{<:DiscreteMixedStrategy})
    return expected_value(x_others -> payoff(Πp, xp, x_others), σ_others)
end
function payoff(Πp::AbstractPayoff, σp::DiscreteMixedStrategy, σ_others::Vector{<:DiscreteMixedStrategy})
    return expected_value(xp -> payoff(Πp, xp, σ_others), σp)
end
# TODO: implement payoff for σp and x_others (mixed for `p`, pure for others)
