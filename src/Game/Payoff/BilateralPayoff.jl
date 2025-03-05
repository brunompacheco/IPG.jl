
"""
A bilateral payoff can be computed as the sum of bilateral interactions between the players.

If all players have bilateral payoffs, an equilibrium for the sampled game can be solved
    using its polymatrix representation, which yields an efficient formulation for the
    optimization problem.

It is expected that each bilateral payoff type implements the `bilateral_payoff` function
    with two methods. The first must compute the independent payoff of a given strategy, and
    must have the following signature:
```julia
bilateral_payoff(Πp::MyBilateralPayoff, xp::Vector{<:Any})
```
The second method must implement the payoff of a strategy with respect to the strategy of another player (`k`).
    It must have the following signature:
```julia
bilateral_payoff(Πp::MyBilateralPayoff, xp::Vector{<:Union{Real,VariableRef}}, xk::Vector{<:Real}, k::Integer)
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
    p::Integer
    function QuadraticPayoff(cp::Vector{<:Real}, Qp::Vector{<:Matrix{<:Real}}, p::Integer)
        if !all(Qpk->size(Qpk,2)==length(cp), Qp)
            error("All Qp matrices must have the same number of columns as elements in cp (= dimension of p's strategy space).")
        end
        return new(cp, Qp, p)
    end
end
function QuadraticPayoff(cp::Real, Qp::Vector{<:Real}, p::Integer)
    # constructor for simpler, unidimensional cases
    return QuadraticPayoff([cp], [qpk * ones(1,1) for qpk in Qp], p)
end

"Compute the individual payoff (independent utility) of strategy `xp`."
function bilateral_payoff(Πp::QuadraticPayoff, xp::Vector{<:Any})
    return Πp.cp' * xp - 0.5 * xp' * Πp.Qp[Πp.p] * xp
end
"Compute each component of the payoff with respect to player `k`."
function bilateral_payoff(Πp::QuadraticPayoff, xp::Vector{<:Any}, xk::Vector{<:Any}, k::Integer)
    return xk' * Πp.Qp[k] * xp
end


# Utils

function bilateral_payoff(Πp::AbstractBilateralPayoff, σp::DiscreteMixedStrategy)
    return expected_value(xp -> bilateral_payoff(Πp, xp), σp)
end
function bilateral_payoff(Πp::AbstractBilateralPayoff, xp::Vector{<:Any}, σk::DiscreteMixedStrategy, k::Integer)
    return expected_value(xk -> bilateral_payoff(Πp, xp, xk, k), σk)
end
function bilateral_payoff(Πp::AbstractBilateralPayoff, σp::DiscreteMixedStrategy, xk::Vector{<:Any}, k::Integer)
    return expected_value(xp -> bilateral_payoff(Πp, xp, xk, k), σp)
end
function bilateral_payoff(Πp::AbstractBilateralPayoff, σp::DiscreteMixedStrategy, σk::DiscreteMixedStrategy, k::Integer)
    return expected_value(xp -> bilateral_payoff(Πp, xp, σk, k), σp)
end

"Compute the payoff of playing `xp` given that the other players play `x_others`."
function payoff(Πp::AbstractBilateralPayoff, xp::Vector{<:Any}, x_others::Vector{<:Vector{<:Any}})
    payoff_value = bilateral_payoff(Πp, xp)
    for k in 1:(length(x_others)+1)
        if k < Πp.p
            payoff_value += bilateral_payoff(Πp, xp, x_others[k], k)
        elseif k > Πp.p
            payoff_value += bilateral_payoff(Πp, xp, x_others[k-1], k)
        end
    end
    return payoff_value
end
