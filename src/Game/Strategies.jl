using IterTools


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
    for (probs, x) in zip(product([σp.probs for σp in σ]...), product([σp.supp for σp in σ]...))
        prob = prod(probs)
        x = collect(x)  # convert tuple to vector
        expectation += prob * f(x)
    end

    return expectation
end
