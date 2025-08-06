
"Strategy profile. We expect `T` to be a DiscreteMixedStrategy or a pure strategy (Vector{<:Real})."
const Profile{T} = Dict{Player, T} where T <: Strategy
export Profile

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

function others(profile::Profile{T}, player::Player) where T <: Strategy
    return Profile{T}(p => profile[p] for p in keys(profile) if p != player)
end
