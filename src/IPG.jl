module IPG

using NormalGames


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

"Compute the payoff of player `p` given strategies x."
function payoff(Πp::QuadraticPayoff, x::Vector{<:Vector{<:Real}}, p::Integer)
    return (
        Πp.cp' * x[p]
        - 0.5 * x[p]' * Πp.Qp[p] * x[p]
        + sum([x[k]' * Πp.Qp[k] * x[p] for k in 1:length(x) if k != p])
    )
end
function payoff(Πp::QuadraticPayoff, x::Vector{<:Real}, p::Integer)
    return payoff(Πp, [[xp] for xp in x], p)
end

struct Player
    Xp::Strategies
    Πp::QuadraticPayoff
end
"Number of variables of player p."
function np(P::Player)
    return np(P.Xp)
end
function np(Xp::Strategies)
    return size(Xp.Ap,1)
end

struct Game
    M::Vector{Player}
end


end # module IPG
