
using LinearAlgebra

const Polymatrix = Dict{Tuple{Player, Player}, Matrix{Float64}}

"Compute the component of the payoff that doesn't depend on other players."
function compute_self_payoff(p::Player, x_p::PureStrategy)
    var_assignments_p = build_var_assignments(p, x_p)

    self_linear_payoff = sum(get(p.Π.aff.terms, v, 0) * var_assignments_p[v] for v in all_variables(p))
    # note the get() may be necessary as there may not be terms for all variables
    self_affine_payoff = p.Π.aff.constant + self_linear_payoff
    # TODO: maybe Dict{VariableRef, Number} should be the standard for assignments
    self_quad_payoff = sum(get(p.Π.terms, UnorderedPair(v,v), 0) * var_assignments_p[v]^2 for v in all_variables(p))

    return self_affine_payoff + self_quad_payoff
end

function compute_bilateral_payoff(p::Player, x_p::PureStrategy, k::Player, x_k::PureStrategy)
    var_assignments_k = build_var_assignments(k, x_k)
    var_assignments_p = build_var_assignments(p, x_p)  # TODO: this could be cached.
    # In fact, +1 for having Dict{VariableRef, Number} as the standard for assignments

    return sum(
        get(p.Π.terms, UnorderedPair(vp,vk), 0) * var_assignments_p[vp] * var_assignments_k[vk]
        for vp in all_variables(p), vk in all_variables(k)
    )
end

"Compute polymatrix for normal form game from sample of strategies."
function get_polymatrix(players::Vector{Player}, S_X::Dict{Player, Vector{PureStrategy}})::Polymatrix
    if length(players) == 2
        # if there are only two players, we can handle generic payoffs
        return get_polymatrix_twoplayers(players[1], players[2], S_X)
    else
        # otherwise, we need payoffs to be bilateral
        return get_polymatrix_bilateral(players, S_X)
    end
end

function get_polymatrix_bilateral(players::Vector{Player}, S_X::Dict{Player, Vector{PureStrategy}})::Polymatrix
    # TODO: we could have the payoff type as a Player parameter, so that we can filter that out straight away
    polymatrix = Polymatrix()

    # compute utility of each player `p` using strategy `i_p` against player `k` using strategy `i_k`
    for p in players
        for k in players
            polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))

            # I am distributing the "self-payoff" through all players, as it seems to be the
            # way NormalGames.jl expects it
            if k != p
                for (i_p, x_p) in enumerate(S_X[p])
                    self_payoff = compute_self_payoff(p, x_p)

                    for (i_k, x_k) in enumerate(S_X[k])
                        polymatrix[p,k][i_p,i_k] = self_payoff + compute_bilateral_payoff(p, x_p, k, x_k)
                    end
                end
            end
        end
    end

    return polymatrix
end

"Compute polymatrix between players `p` and `k` in a two-player game."
function get_polymatrix_twoplayers(p::Player, k::Player, S_X::Dict{Player, Vector{PureStrategy}})::Polymatrix
    # initialization
    polymatrix = Polymatrix()

    polymatrix[p,p] = zeros(length(S_X[p]), length(S_X[p]))
    polymatrix[k,k] = zeros(length(S_X[k]), length(S_X[k]))

    polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))
    polymatrix[k,p] = zeros(length(S_X[k]), length(S_X[p]))

    # compute polymatrix for two-player game
    for i_p in eachindex(S_X[p])
        for i_k in eachindex(S_X[k])
            polymatrix[p,k][i_p,i_k] = payoff(p, S_X[p][i_p], Dict(k => S_X[k][i_k]))
            polymatrix[k,p][i_k,i_p] = payoff(k, S_X[k][i_k], Dict(p => S_X[p][i_p]))
        end
    end

    return polymatrix
end

"We expect the new strategies to always be the last ones in S_X[p]."
function update_polymatrix!(polymatrix::Polymatrix, p::Player, S_X::Dict{Player, Vector{PureStrategy}})
    other_players = collect(filter(k -> k != p, keys(S_X)))
    sub_S_X = Dict{Player, Vector{PureStrategy}}(k => S_X[k] for k in other_players)
    n_old_p_strats = size(polymatrix[p,other_players[1]],1)
    sub_S_X[p] = S_X[p][(n_old_p_strats+1):end]  # only the new strategies

    # we compute a new polymatrix considering only the new strategies added for player `p`
    new_strats_polymatrix = get_polymatrix(collect(keys(S_X)), sub_S_X)

    # then, it's just a matter of stitching the new polymatrix to the old one
    for k in other_players
        polymatrix[p,k] = [polymatrix[p,k] ; new_strats_polymatrix[p,k]]
        polymatrix[k,p] = [polymatrix[k,p] new_strats_polymatrix[k,p]]
    end
    polymatrix[p,p] = Diagonal([diag(polymatrix[p,p]) ; diag(new_strats_polymatrix[p,p])])
end
