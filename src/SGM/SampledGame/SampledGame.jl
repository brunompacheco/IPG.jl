
"Compute polymatrix for normal form game from sample of strategies."
function get_polymatrix(players::Vector{<:Player{<:AbstractBilateralPayoff}}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    polymatrix = Dict{Tuple{Integer, Integer}, Matrix{Float64}}()

    # compute utility of each player `p` using strategy `i_p` against player `k` using strategy `i_k`
    for player_p in players
        for player_k in players
            p = player_p.p
            k = player_k.p

            polymatrix[p,k] = zeros(length(S_X[p]), length(S_X[k]))

            if k != p
                for i_p in 1:length(S_X[p])
                    for i_k in 1:length(S_X[k])
                        polymatrix[p,k][i_p,i_k] = (
                            IPG.bilateral_payoff(players[p].Πp, p, S_X[p][i_p], k, S_X[k][i_k])
                            + IPG.bilateral_payoff(players[p].Πp, p, S_X[p][i_p], p, S_X[p][i_p])
                        )
                    end
                end
            end
        end
    end

    return polymatrix
end

abstract type AbstractSampledGame end

mutable struct SampledGame <: AbstractSampledGame
    S_X::Vector{Vector{Vector{Float64}}}  # sample of strategies (finite subset of the strategy space X)
end

"Normal-form polymatrix representation of the sampled game."
mutable struct PolymatrixSampledGame <: AbstractSampledGame
    S_X::Vector{Vector{Vector{Float64}}}  # sample of strategies (finite subset of the strategy space X)
    polymatrix::Dict{Tuple{Int, Int}, Matrix{Float64}}
end
function SampledGame(players::Vector{<:AbstractPlayer}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    error("Not implemented yet.")

    # if length(players) == 2
    #     return PolymatrixSampledGame(S_X, get_polymatrix(players, S_X))
    # else
    #     return SampledGame(S_X)
    # end
end
function SampledGame(players::Vector{<:Player{<:AbstractBilateralPayoff}}, S_X::Vector{<:Vector{<:Vector{<:Real}}})
    return PolymatrixSampledGame(S_X, get_polymatrix(players, S_X))
end

function add_new_strategy!(sg::AbstractSampledGame, players::Vector{<:Player{<:AbstractBilateralPayoff}}, new_xp::Vector{<:Real}, p::Integer)
    push!(sg.S_X[p], new_xp)
end

function add_new_strategy!(sg::PolymatrixSampledGame, players::Vector{<:Player{<:AbstractBilateralPayoff}}, new_xp::Vector{<:Real}, p::Integer)
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
                    IPG.bilateral_payoff(players[p1].Πp, p1, sg.S_X[p1][i], p, new_xp)
                    + IPG.bilateral_payoff(players[p1].Πp, p1, sg.S_X[p1][i], p1, sg.S_X[p1][i])
                )
            end
        elseif (p1 == p) & (p2 != p)
            # add new row to polymatrix to store the utilities wrt the new strategy
            sg.polymatrix[p,p2] = vcat(sg.polymatrix[p,p2], zeros(1, strat[p2]))

            for i in 1:strat[p2]
                # compute utility of player `p1` using strategy `i` against the new strategy of player `p`
                sg.polymatrix[p,p2][end,i] = (
                    IPG.bilateral_payoff(players[p].Πp, p, new_xp, p2, sg.S_X[p2][i])
                    + IPG.bilateral_payoff(players[p].Πp, p, new_xp, p, new_xp)
                )
            end
        end
    end
end

include("SearchNE.jl")
