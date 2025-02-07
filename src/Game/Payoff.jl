using JuMP

abstract type AbstractPayoff end

"Payoff function of player `p` with quadratic bilateral (pairwise) interactions."
struct QuadraticPayoff <: AbstractPayoff
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
    return QuadraticPayoff([cp], [qpk * ones(1,1) for qpk in Qp])
end

"Compute each component of the payoff of player `p` with respect to player `k`."
function bilateral_payoff(Πp::QuadraticPayoff, p::Integer, xp::Vector{<:Union{Real,VariableRef}}, k::Integer, xk::Vector{<:Real})
    if p == k
        return Πp.cp' * xp - 0.5 * xp' * Πp.Qp[p] * xp
    else
        return xk' * Πp.Qp[k] * xp
    end
end
function bilateral_payoff(Πp::QuadraticPayoff, p::Integer, xp::Vector{<:Union{Real,VariableRef}}, k::Integer, σk::DiscreteMixedStrategy)
    return expected_value(xk -> bilateral_payoff(Πp, p, xp, k, xk), σk)
end

"Compute the payoff of player `p` given strategies x."
function payoff(Πp::QuadraticPayoff, x::Vector{<:Vector{<:Real}}, p::Integer)
    return sum([bilateral_payoff(Πp, p, x[p], k, x[k]) for k in 1:length(x)])
end
function payoff(Π::Vector{QuadraticPayoff}, x::Vector{<:Vector{<:Real}}, p::Integer)
    return sum([bilateral_payoff(Π[p], p, x[p], k, x[k]) for k in 1:length(x)])
end
function payoff(Π::Vector{QuadraticPayoff}, σ::Vector{DiscreteMixedStrategy}, p::Integer)
    _payoff = x -> payoff(Π, x, p)
    return expected_value(_payoff, σ)
end