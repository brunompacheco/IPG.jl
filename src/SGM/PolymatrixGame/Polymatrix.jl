
using LinearAlgebra

const Polymatrix = Dict{Tuple{Player, Player}, Matrix{Float64}}


function compute_self_payoff(Π::AffExpr, v_bar::AssignmentDict)::Float64
    # note the get() may be necessary as there may not be terms for all variables
    self_linear_payoff = sum(get(Π.terms, ref, 0) * val for (ref, val) in v_bar)

    # constant is here by convention (inherited from NormalGames.jl)
    return Π.constant + self_linear_payoff
end

function compute_self_payoff(Π::QuadExpr, v_bar::AssignmentDict)::Float64
    # TODO: maybe Dict{VariableRef, Number} should be the standard for assignments
    self_quad_payoff = sum(get(Π.terms, UnorderedPair(ref,ref), 0) * val^2 for (ref, val) in v_bar)

    return self_quad_payoff + compute_self_payoff(Π.aff, v_bar)
end

"Compute the component of the payoff that doesn't depend on other players."
function compute_self_payoff(p::Player, x_p::PureStrategy)
    v_bar = Assignment(p, x_p)

    return compute_self_payoff(p.Π, v_bar)
end

"Compute player p's payoff component from some *other* player playing v_bar_k."
function compute_others_payoff(Π::AffExpr, v_bar_k::AssignmentDict)::Float64
    # TODO: identical to compute_self_payoff(::AffExpr, ::AssignmentDict), except for the constant. I could refactor
    return sum(  # terms of the form q_j * xk_j, where xk_j belongs to player k
        get(Π.terms, ref, 0) * val for (ref, val) in v_bar_k
    )
end

"Compute player p's payoff component from her playing v_bar_p and some *other* player playing v_bar_k."
function compute_bilateral_payoff(Π::QuadExpr, v_bar_p::AssignmentDict, v_bar_k::AssignmentDict)::Float64
    mixed_components = sum(  # terms of the form q_ij * xp_i * xk_j, where xp_i belongs to player p and xk_j belongs to player k
        get(Π.terms, UnorderedPair(ref_i,ref_j), 0) * val_i * val_j
        for (ref_i, val_i) in v_bar_p
        for (ref_j, val_j) in v_bar_k
    )
    other_components = sum(  # terms of the form q_ij * xk_i * xk_j, where xk_i,xk_j belong to player k
        get(Π.terms, UnorderedPair(ref_i,ref_j), 0) * val_i * val_j
        for (ref_i, val_i) in v_bar_k
        for (ref_j, val_j) in v_bar_k
    )
    other_components = other_components / 2  # I'm iterating over al possible unordered pairs twice!

    return mixed_components + other_components + compute_others_payoff(Π.aff, v_bar_k)
end

function compute_bilateral_payoff(p::Player, x_p::PureStrategy, k::Player, x_k::PureStrategy)
    # In fact, +1 for having Dict{VariableRef, Number} as the standard for assignments
    v_bar_p = Assignment(p, x_p)  # TODO: this could be cached.
    v_bar_k = _internalize_assignment(p, Assignment(k, x_k))

    return compute_bilateral_payoff(p.Π, v_bar_p, v_bar_k)
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
