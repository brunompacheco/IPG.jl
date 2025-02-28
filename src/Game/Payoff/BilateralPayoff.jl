
"""
A bilateral payoff can be computed as the sum of bilateral interactions between the players.

If all players have bilateral payoffs, an equilibrium for the sampled game can be solved
using its polymatrix representation, which yields an efficient formulation for the
optimization problem.

It is expected that each bilateral payoff type implements the `bilateral_payoff` function.
The function must have the following signature:
```julia
function bilateral_payoff(Πp::MyBilateralPayoff, p::Integer, xp::Vector{<:Union{Real,VariableRef}}, k::Integer, xk::Vector{<:Real})
    [...]
end
```
Note that the `xp` argument can also be a vector of JuMP variable references. This is
because the payoff functions are also used to build JuMP expressions, e.g., in
`best_response` (see `src/Game/Player.jl`).
"""
abstract type AbstractBilateralPayoff <: AbstractPayoff end

"Payoff function of player `p` with quadratic bilateral (pairwise) interactions."
struct QuadraticPayoff <: AbstractBilateralPayoff
    cp::Vector{Float64}
    "`x[k]' * Qp[k] * x[p]` is the payoff component for player `p` with respect to the strategy of player `k`."
    Qp::Vector{Matrix{Float64}}
    function QuadraticPayoff(cp::Vector{<:Real}, Qp::Vector{<:Matrix{<:Real}})
        if !all(Qpk->size(Qpk,2)==length(cp), Qp)
            error("All Qp matrices must have the same number of columns as elements in cp (= dimension of p's strategy space).")
        end
        return new(cp, Qp)
    end
end
function QuadraticPayoff(cp::Real, Qp::Vector{<:Real})
    # constructor for simpler, unidimensional cases
    return QuadraticPayoff([cp], [qpk * ones(1,1) for qpk in Qp])
end

"Compute each component of the payoff of player `p` with respect to player `k`."
function bilateral_payoff(Πp::QuadraticPayoff, p::Integer, xp::Vector{<:Any}, k::Integer, xk::Vector{<:Any})
    if p == k
        return Πp.cp' * xp - 0.5 * xp' * Πp.Qp[p] * xp
    else
        return xk' * Πp.Qp[k] * xp
    end
end


# Utils

function bilateral_payoff(Πp::AbstractBilateralPayoff, p::Integer, xp::Vector{<:Any}, k::Integer, σk::DiscreteMixedStrategy)
    return expected_value(xk -> bilateral_payoff(Πp, p, xp, k, xk), σk)
end
function bilateral_payoff(Πp::AbstractBilateralPayoff, p::Integer, σp::DiscreteMixedStrategy, k::Integer, σk::DiscreteMixedStrategy)
    return expected_value(xp -> bilateral_payoff(Πp, p, xp, k, σk), σp)
end

"Compute the payoff of player `p` given strategies x."
function payoff(Πp::AbstractBilateralPayoff, x::Vector{<:Vector{<:Any}}, p::Integer)
    payoff_value = bilateral_payoff(Πp, p, x[p], 1, x[1])
    for k in 2:length(x)
        payoff_value += bilateral_payoff(Πp, p, x[p], k, x[k])
    end
    return payoff_value
end
