
"Strategy profile. We expect `T` to be a DiscreteMixedStrategy or a pure strategy (Vector{<:Real})."
const Profile{T} = Dict{Player, T} where T <: Strategy
export Profile

"Compute the expected value of a function given a discrete mixed profile."
function expected_value(f::Function, σ::Profile{DiscreteMixedStrategy})
    expectation = 0

    # iterate over all possible *pure* strategy profiles
    for (prob, pure_profile) in zip(probabilities(σ), support(σ))
        expectation += prob * f(pure_profile)
    end

    return expectation
end

function support(σ::Profile{DiscreteMixedStrategy})
    # TODO: make this iterable? it would be nice to have lazy evaluation here...
    return [Dict(zip(keys(σ), x)) for x in Iterators.product([σp.supp for σp in values(σ)]...)]
end

function probabilities(σ::Profile{DiscreteMixedStrategy})
    # TODO: make this iterable? it would be nice to have lazy evaluation here...
    return [prod(probs) for probs in Iterators.product([σp.probs for σp in values(σ)]...)]
end

function others(profile::Profile{T}, player::Player) where T <: Strategy
    return Profile{T}(p => profile[p] for p in keys(profile) if p != player)
end
