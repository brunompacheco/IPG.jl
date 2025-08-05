
"Strategy profile. We expect `T` to be a DiscreteMixedStrategy or a pure strategy (Vector{<:Real})."
const Profile{T} = Dict{Player, T}

"Compute the expected value of a function given a discrete mixed profile."
function expected_value(f::Function, σ::Profile{DiscreteMixedStrategy})
    expectation = 0

    # iterate over all possible *pure* strategy profiles
    for (probs, x) in zip(Iterators.product([σp.probs for σp in values(σ)]...), Iterators.product([σp.supp for σp in values(σ)]...))
        prob = prod(probs)
        pure_profile = Dict(zip(keys(σ), x))  # create a pure profile
        expectation += prob * f(pure_profile)
    end

    return expectation
end
