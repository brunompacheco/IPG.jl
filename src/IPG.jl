module IPG

using NormalGames

export Strategies, np, QuadraticPayoff, payoff, Game


"Each player's set of strategies is defined by `Ap * xp <= bp`."
struct Strategies
    Ap::Matrix{Float64}
    bp::Vector{Float64}
    "xp[1],...,xp[Bp] are integer variables."
    Bp::Integer
end
# TODO: WARNING: Method definition (::Type{IPG.Strategies})(Array{Float64, 2}, Array{Float64, 1}, Int64) in module IPG at /workspaces/IPG/src/IPG.jl:8 overwritten at /workspaces/IPG/src/IPG.jl:13.
# TODO: ERROR: Method overwriting is not permitted during Module precompilation. Use `__precompile__(false)` to opt-out of precompilation.
# function Strategies(Ap::Matrix{Float64}, bp::Vector{Float64}, Bp::Int64)
#     @assert size(Ap,1) == length(bp), "Ap and bp must have the same number of rows."
#     @assert Bp <= size(Ap,2), "Bp must be less than or equal to the number of columns of Ap."
#     return Strategies(Ap, bp, Bp)
# end
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
end
function QuadraticPayoff(cp::Real, Qp::Vector{<:Real})
    return QuadraticPayoff([cp], [qpk * ones(1,1) for qpk in Qp])
end

"Compute each component of the payoff of player `p` with respect to player `k`."
function bilateral_payoff(Πp::QuadraticPayoff, p::Integer, xp::Vector{<:Real}, k::Integer, xk::Vector{<:Real})
    if p == k
        return Πp.cp' * xp - 0.5 * xp' * Πp.Qp[p] * xp
    else
        return xk' * Πp.Qp[k] * xp
    end
end

"Compute the payoff of player `p` given strategies x."
function payoff(Π::Vector{QuadraticPayoff}, x::Vector{<:Vector{<:Real}}, p::Integer)
    return sum([bilateral_payoff(Π[p], p, x[p], k, x[k]) for k in 1:length(x)])
end

struct Game
    X::Vector{Strategies}
    Π::Vector{QuadraticPayoff}
end

end # module IPG
