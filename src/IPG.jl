module IPG

using NormalGames
using JuMP, IterTools

export Strategies, np, QuadraticPayoff, payoff, Game, DiscreteMixedStrategy, expected_value


struct DiscreteMixedStrategy
    "Probability vector."
    probs::Vector{Float64}
    "Support (vector of strategies)."
    supp::Vector{Vector{Float64}}
    function DiscreteMixedStrategy(probs::Vector{<:Real}, supp::Vector{<:Vector{<:Real}})
        if length(probs) != size(supp, 1)
            error("There must be as many probabilities as strategies in the support.")
        end
        if sum(probs) != 1
            error("Probabilities must sum to 1.")
        end
        if any(probs .< 0)
            error("Probabilities must be non-negative.")
        end

        # remove strategies with zero probability
        return new(probs[probs .> 0], supp[probs .> 0])
    end
end

"Compute the expected value of a function given a discrete mixed strategy."
function expected_value(f::Function, σp::DiscreteMixedStrategy)
    expectation = 0
    for (prob, xp) in zip(σp.probs, σp.supp)
        expectation += prob * f(xp)
    end

    return expectation
end

"Compute the expected value of a function given a discrete mixed profile."
function expected_value(f::Function, σ::Vector{<:DiscreteMixedStrategy})
    expectation = 0

    # iterate over all possible _pure_ strategy profile
    for (probs, x) in zip(product([σp.probs for σp in σ]...), product([σp.supp for σp in σ]...))
        prob = prod(probs)
        x = collect(x)  # convert tuple to vector
        expectation += prob * f(x)
    end

    return expectation
end

"Each player's set of strategies is defined by `Ap * xp <= bp`."
struct Strategies
    Ap::Matrix{Float64}
    bp::Vector{Float64}
    "xp[1],...,xp[Bp] are integer variables."
    Bp::Integer
    function Strategies(Ap::Matrix{<:Real}, bp::Vector{<:Real}, Bp::Integer)
        if size(Ap,1) != length(bp)
            error("Ap and bp must have the same number of rows.")
        end
        if Bp > size(Ap,2)
            error("Bp must be less than or equal to the number of columns of Ap.")
        end
        if Bp < 0
            error("There cannot be a negative number of integer variables.")
        end

        return new(Ap, bp, Bp)
    end
end
function Strategies(ap::Real, bp::Real, Bp::Integer)
    return Strategies(ap * ones(1,1), [bp], Bp)
end
function Strategies(ap::Vector{<:Real}, bp::Real, Bp::Integer)
    return Strategies(ones(1,length(ap)) .* ap, [bp], Bp)
end

function np(Xp::Strategies)
    return size(Xp.Ap,1)
end

# TODO: create an abstract Payoff type
"Payoff function of player `p` with quadratic bilateral (pairwise) interactions."
struct QuadraticPayoff
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
    _bilateral_payoff = xk -> bilateral_payoff(Πp, p, xp, k, xk)
    return expected_value(_bilateral_payoff, σk)
end

"Compute the payoff of player `p` given strategies x."
function payoff(Π::Vector{QuadraticPayoff}, x::Vector{<:Vector{<:Real}}, p::Integer)
    return sum([bilateral_payoff(Π[p], p, x[p], k, x[k]) for k in 1:length(x)])
end
function payoff(Π::Vector{QuadraticPayoff}, σ::Vector{DiscreteMixedStrategy}, p::Integer)
    _payoff = x -> payoff(Π, x, p)
    return expected_value(_payoff, σ)
end

struct Game
    X::Vector{Strategies}
    Π::Vector{QuadraticPayoff}
end

end # module IPG
