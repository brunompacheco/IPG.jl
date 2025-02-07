using JuMP

abstract type AbstractPlayer end

"A player in an IPG."
struct Player <: AbstractPlayer
    "Strategy space."
    Xp::Model
    "Payoff function."
    Πp::AbstractPayoff
    "Player's index."
    p::Integer  # TODO: this could be Any, to allow for more general collections, e.g, string names
    # TODO: maybe I could just index everything relevant as a Dict{Player, T}?
end
"Initialize player with empty strategy space."
function Player(Πp::AbstractPayoff, p::Integer)
    return Player(Model(), Πp, p)
end

"Compute the utility that `player_p` receives from `player_k` when they play, resp., `xp` and `xk`."
function bilateral_payoff(player_p::Player, xp::Vector{<:Union{Real,VariableRef}}, player_k::Player, xk::Vector{<:Real})
    return bilateral_payoff(player_p.Πp, player_p.p, xp, player_k.p, xk)
end
"Compute the utility that `player_p` receives from `player_k` when they play, resp., `xp` and `σk`."
function bilateral_payoff(player_p::Player, xp::Vector{<:Union{Real,VariableRef}}, player_k::Player, σk::DiscreteMixedStrategy)
    return expected_value(xk -> bilateral_payoff(player_p.Πp, player_p.p, xp, player_k.p, xk), σk)
end

"Compute the payoff of player `player` given pure strategy profile `x`."
function payoff(player::Player, x::Vector{<:Vector{<:Real}})
    return payoff(player.Πp, x, player.p)
end
"Compute the payoff of player `player` given mixed strategy profile `σ`."
function payoff(player::Player, σ::Vector{DiscreteMixedStrategy})
    _payoff = x -> payoff(player, x)
    return expected_value(_payoff, σ)
end

"Solve the feasibility problem for a player, returning a feasible strategy."
function find_feasible_pure_strategy(player::Player, optimizer_factory=nothing)
    if ~isnothing(optimizer_factory)
        set_optimizer(player.Xp, optimizer_factory)
    end

    # it is simply a feasibility problem
    @objective(player.Xp, Min, 0)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(all_variables(player.Xp))
end

"Solve the feasibility problem of all players, returning a feasible profile."
function find_feasible_pure_profile(players::Vector{Player}, optimizer_factory=nothing)
    return [find_feasible_pure_strategy(player, optimizer_factory) for player in players]
end