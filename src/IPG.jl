module IPG

using NormalGames

"Each player's set of strategies is defined by `Ap * xp <= bp`."
struct PlayerStrategies
    Ap::Matrix{Float64}
    bp::Vector{Float64}
    "xp[1],...,xp[Bp] are integer variables."
    Bp::Int64
end
"Number of variables of player p."
function np(Xp::PlayerStrategies)
    return size(Xp.Ap,1)
end

"Separable quadratic payoff functions with bilateral (pairwise) interactions"
struct QuadraticPayoff
    c::Vector{Vector{Float64}}
    Q::Dict{Tuple{Int64, Int64}, Matrix{Float64}}
end
"Compute the payoff of player p given strategies x."
function payoff(Π::QuadraticPayoff, x::Vector{Vector{Float64}}, p::Int64)
    return (
        Π.c[p] * x[p]
        - 0.5 * x[p]' * Π.Q[p,p] * x[p]
        + sum([x[k]' * Π.Q[p,k] * x[p] for k in 1:length(x) if k != p])
    )
end


end # module IPG
