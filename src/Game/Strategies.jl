using IterTools


struct DiscreteMixedStrategy
    "Probability vector."
    probs::Vector{Float64}
    "Support (vector of strategies)."
    supp::Vector{<:Any}
    function DiscreteMixedStrategy(probs::Vector{<:Real}, supp::Vector{<:Any})
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
"Build mixed strategy from pure strategy."
function DiscreteMixedStrategy(xp::Vector{<:Any})
    return DiscreteMixedStrategy([1.0], [xp])
end
Base.:(==)(σp1::DiscreteMixedStrategy, σp2::DiscreteMixedStrategy) = σp1.probs == σp2.probs && σp1.supp == σp2.supp
Base.:(≈)(σp1::DiscreteMixedStrategy, σp2::DiscreteMixedStrategy) = σp1.probs ≈ σp2.probs && σp1.supp ≈ σp2.supp

"Check whether the mixed strategy is actually pure."
function is_pure(σp::DiscreteMixedStrategy)
    return length(σp.probs) == 1
end

"Compute the expected value of a function given a discrete mixed strategy."
function expected_value(f::Function, σp::DiscreteMixedStrategy)
    expectation = 0 .* f(σp.supp[1])  # guess the output type of f
    for (prob, xp) in zip(σp.probs, σp.supp)
        expectation += prob .* f(xp)
    end

    return expectation
end

"Compute the expected value of a function given a discrete mixed profile."
function expected_value(f::Function, σ::Vector{<:DiscreteMixedStrategy})
    expectation = 0

    # iterate over all possible _pure_ strategy profile
    for (probs, x) in zip(Iterators.product([σp.probs for σp in σ]...), Iterators.product([σp.supp for σp in σ]...))
        prob = prod(probs)
        x = collect(x)  # convert tuple to vector
        expectation += prob * f(x)
    end

    return expectation
end

# Utils

function others(x::Vector{<:Any}, p::Integer)
    return [x[1:p-1] ; x[p+1:end]]
end
