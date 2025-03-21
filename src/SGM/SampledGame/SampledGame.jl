
abstract type AbstractSampledGame end

mutable struct GenericSampledGame <: AbstractSampledGame
    S_X::Vector{<:Vector{<:PureStrategy}}  # sample of strategies (finite subset of the strategy space X)
end
function GenericSampledGame(players::Vector{<:AbstractPlayer}, S_X::Vector{<:Vector{<:PureStrategy}})
    error("Not implemented yet.")

    # if length(players) == 2
    #     return PolymatrixSampledGame(S_X, get_polymatrix(players, S_X))
    # else
    #     return SampledGame(S_X)
    # end
end

function add_new_strategy!(sg::GenericSampledGame, players::Vector{<:AbstractPlayer}, new_xp::PureStrategy, p::Integer)
    push!(sg.S_X[p], new_xp)
end

"Compute polymatrix for normal form game from sample of strategies."
function get_polymatrix(players::Vector{<:Player{<:AbstractBilateralPayoff}}, S_X::Vector{<:Vector{<:PureStrategy}})
    polymatrix = Dict{Tuple{Integer, Integer}, Matrix{Float64}}()

    # compute utility of each player `p` using strategy `i_p` against player `k` using strategy `i_k`
    for player_p in players
        for player_k in players
            p = player_p.p
            k = player_k.p

            polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))

            if k != p
                for i_p in eachindex(S_X[p])
                    for i_k in eachindex(S_X[k])
                        polymatrix[p,k][i_p,i_k] = (
                            IPG.bilateral_payoff(players[p].Πp, S_X[p][i_p], S_X[k][i_k], k)
                            + IPG.bilateral_payoff(players[p].Πp, S_X[p][i_p])
                        )
                    end
                end
            end
        end
    end

    return polymatrix
end

function get_polymatrix(player1::AbstractPlayer, player2::AbstractPlayer, S_X::Vector{<:Vector{<:PureStrategy}})
    p = player1.p
    k = player2.p

    # initialization
    polymatrix = Dict{Tuple{Integer, Integer}, Matrix{Float64}}()

    polymatrix[p,p] = zeros(length(S_X[p]), length(S_X[p]))
    polymatrix[k,k] = zeros(length(S_X[k]), length(S_X[k]))

    polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))
    polymatrix[k,p] = zeros(length(S_X[k]), length(S_X[p]))

    # compute polymatrix for two-player game
    for i_p in eachindex(S_X[p])
        for i_k in eachindex(S_X[k])
            polymatrix[p,k][i_p,i_k] = payoff(player1.Πp, S_X[p][i_p], [S_X[k][i_k]])
            polymatrix[k,p][i_k,i_p] = payoff(player2.Πp, S_X[k][i_k], [S_X[p][i_p]])
        end
    end

    return polymatrix
end

"Normal-form polymatrix representation of the sampled game."
mutable struct PolymatrixSampledGame <: AbstractSampledGame
    S_X::Vector{<:Vector{<:PureStrategy}}  # sample of strategies (finite subset of the strategy space X)
    polymatrix::Dict{Tuple{Int, Int}, Matrix{Float64}}
end

function PolymatrixSampledGame(players::Vector{<:Player{<:AbstractBilateralPayoff}}, S_X::Vector{<:Vector{<:PureStrategy}})
    return PolymatrixSampledGame(S_X, get_polymatrix(players, S_X))
end
function PolymatrixSampledGame(players::Vector{<:AbstractPlayer}, S_X::Vector{<:Vector{<:PureStrategy}})
    @assert length(players) == 2  "Cannot build polymatrix for more than two players unless their payoffs are bilatera (see `BilateralPayoff`)"

    return PolymatrixSampledGame(S_X, get_polymatrix(players[1], players[2], S_X))
end

function add_new_strategy!(sg::PolymatrixSampledGame, players::Vector{<:Player{<:AbstractBilateralPayoff}}, new_xp::PureStrategy, p::Integer)
    # first part is easy, just add the new strategy to the set
    push!(sg.S_X[p], new_xp)
    strat = length.(sg.S_X)

    # now we need to update the normal game (polymatrix)
    for (p1, p2) in keys(sg.polymatrix)
        if p1 == p2 == p
            # add new row to polymatrix to store the utilities wrt the new strategy
            sg.polymatrix[p,p] = zeros(strat[p], strat[p])
        elseif (p1 != p) & (p2 == p)
            # add new column to polymatrix to store the utilities wrt the new strategy
            sg.polymatrix[p1,p] = hcat(sg.polymatrix[p1,p], zeros(strat[p1], 1))

            for i in 1:strat[p1]
                # compute utility of player `p1` using strategy `i` against the new strategy of player `p`
                sg.polymatrix[p1,p][i,end] = (
                    IPG.bilateral_payoff(players[p1].Πp, sg.S_X[p1][i], new_xp, p)
                    + IPG.bilateral_payoff(players[p1].Πp, sg.S_X[p1][i])
                )
            end
        elseif (p1 == p) & (p2 != p)
            # add new row to polymatrix to store the utilities wrt the new strategy
            sg.polymatrix[p,p2] = vcat(sg.polymatrix[p,p2], zeros(1, strat[p2]))

            for i in 1:strat[p2]
                # compute utility of player `p1` using strategy `i` against the new strategy of player `p`
                sg.polymatrix[p,p2][end,i] = (
                    IPG.bilateral_payoff(players[p].Πp, new_xp, sg.S_X[p2][i], p2)
                    + IPG.bilateral_payoff(players[p].Πp, new_xp)
                )
            end
        end
    end
end
function add_new_strategy!(sg::PolymatrixSampledGame, players::Vector{<:AbstractPlayer}, new_xp::PureStrategy, p::Integer)
    # this should be just a sanity check, a PolymatrixSampledGame should never be built if
    # the game is being played by more than two players
    @assert length(players) == 2  "Cannot build polymatrix for more than two players unless their payoffs are bilatera (see `BilateralPayoff`)"

    k = p == 1 ? 2 : 1

    player_p = players[p]
    player_k = players[k]

    # first part is easy, just add the new strategy to the set
    push!(sg.S_X[p], new_xp)

    # now we need to update the polymatrix

    # add new row to polymatrix to store the utilities wrt the new strategy
    sg.polymatrix[p,p] = zeros(length(sg.S_X[p]), length(sg.S_X[p]))

    # add new row/column to polymatrix to store the utilities wrt the new strategy
    sg.polymatrix[k,p] = hcat(sg.polymatrix[k,p], zeros(length(sg.S_X[k]), 1))
    sg.polymatrix[p,k] = vcat(sg.polymatrix[p,k], zeros(1, length(sg.S_X[k])))

    for i in eachindex(sg.S_X[k])
        # compute new utilities of player `k` against the new strategy of player `p`
        sg.polymatrix[k,p][i,end] = payoff(player_k.Πp, sg.S_X[k][i], [new_xp])

        # compute new utilities of player `p` using the new strategy against the old
        # strategies of player `k`
        sg.polymatrix[p,k][end,i] = payoff(player_p.Πp, new_xp, [sg.S_X[k][i]])
    end
end

SampledGame = PolymatrixSampledGame  # default value

include("SearchNE.jl")
