
# TODO: the following is a great example for why Sample{T} has to be an OrderedDict if we
# want to stop passing players to this type of function
function get_utilities(players::Vector{Player}, S_X::Sample{PureStrategy})::Array{Float64}
    n_strategies = [length(S_X[p]) for p in players]
    utilities = zeros(Float64, n_strategies..., length(players))

    # iterate over each possible profile in the sample
    for s_idx in Iterators.product([1:s for s in n_strategies]...)
        profile_i = Profile{PureStrategy}(
            p => S_X[p][s_idx[i]] for (i, p) in enumerate(players)
        )

        # compute utility each player gets from the profile
        utilities[s_idx..., :] = [
            payoff(p, profile_i[p], others(profile_i, p)) for p in players
        ]
    end

    return utilities
end

"We expect the new strategies to always be the last ones in S_X."
function update_utilities!(utilities::Array{Float64}, players::Vector{Player}, S_X::Sample{PureStrategy})
    for i in 1:length(players)
        n_old_p_strats = size(utilities, i)
        n_old_p_strats == length(S_X[players[i]]) && continue # no new strategies for this player

        # player with new strategies
        p = players[i]

        # copy of S_X in which we want to have all strategies of the other players...
        sub_S_X = Sample{PureStrategy}(k => S_X[k] for k in others(players, p))
        sub_S_X[p] = S_X[p][(n_old_p_strats+1):end]  # ...and the new strategies of `p`

        new_utilities = get_utilities!(players, sub_S_X)

        utilities = cat(utilities, new_utilities; dims=i)
    end
end
