using IterTools

# TODO: generalize to any numeric type `const PureStrategy{T} = Vector{T} where T <: Real`
const PureStrategy = Vector{Float64}

struct DiscreteMixedStrategy
    "Probability vector."
    probs::Vector{Float64}
    "Support (vector of strategies)."
    supp::Vector{PureStrategy}
    function DiscreteMixedStrategy(probs::Vector{Float64}, supp::Vector{PureStrategy})
        if length(probs) != size(supp, 1)
            error("There must be as many probabilities as strategies in the support.")
        end
        if ~(sum(probs) ≈ 1)
            error("Probabilities must sum to 1.")
        end
        if any(probs .< 0)
            error("Probabilities must be non-negative.")
        end

        # remove strategies with zero probability
        return new(probs[probs .> 0], supp[probs .> 0])
    end
end
Base.convert(::Type{DiscreteMixedStrategy}, xp::PureStrategy) = DiscreteMixedStrategy([1.0], [xp])
Base.:(==)(σp1::DiscreteMixedStrategy, σp2::DiscreteMixedStrategy) = σp1.probs == σp2.probs && σp1.supp == σp2.supp
Base.:(≈)(σp1::DiscreteMixedStrategy, σp2::DiscreteMixedStrategy) = σp1.probs ≈ σp2.probs && σp1.supp ≈ σp2.supp

"Compute the expected value of a function given a discrete mixed strategy."
function expected_value(f::Function, σp::DiscreteMixedStrategy)
    expectation = 0 .* f(σp.supp[1])  # guess the output type of f
    for (prob, xp) in zip(σp.probs, σp.supp)
        expectation += prob .* f(xp)
    end

    return expectation
end
export expected_value

const Strategy = Union{PureStrategy, DiscreteMixedStrategy}
export PureStrategy, DiscreteMixedStrategy, Strategy

# Utils

function others(x::Vector{<:Any}, p::Integer)
    return [x[1:p-1] ; x[p+1:end]]
end
export others
