using JuMP

"""
A payoff function must implement some form of payoff computation for each player in a game.

A custom implementation can have any fields, but it must implement the `payoff` function.
The function must have the following signature:
```julia
function payoff(Πp::MyPayoff, x::Vector{<:Vector{<:Union{Real,VariableRef}}}, p::Integer)
    [...]
end
```
Note that the `xp` argument can also be a vector of JuMP variable references. This is
because the payoff functions are also used to build JuMP expressions, e.g., in
`best_response` (see `src/Game/Player.jl`).
"""
abstract type AbstractPayoff end

struct BlackBoxPayoff <: AbstractPayoff
    "The arguments must be a strategy profile and a player index, and return the player's payoff."
    f::Function
end

"Compute the payoff of player `p` given strategies x."
function payoff(Πp::BlackBoxPayoff, x::Vector{<:Vector{<:Any}}, p::Integer)
    # TODO: this notation is really weird. I think it would be better to always have `x_p`
    # and `x_others` instead of passing indices
    return Πp.f(x, p)
end

include("BilateralPayoff.jl")

# Utils

"Compute expected payoff for player `p` given mixed strategy profile `σ`."
function payoff(Πp::AbstractPayoff, σ::Vector{DiscreteMixedStrategy}, p::Integer)
    return expected_value(x -> payoff(Πp, x, p), σ)
end
