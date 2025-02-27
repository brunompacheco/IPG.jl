using JuMP

abstract type AbstractPayoff end

include("BilateralPayoff.jl")


# Utils

"Compute expected payoff for player `p` given mixed strategy profile `σ`."
function payoff(Πp::AbstractPayoff, σ::Vector{DiscreteMixedStrategy}, p::Integer)
    return expected_value(x -> payoff(Πp, x, p), σ)
end
