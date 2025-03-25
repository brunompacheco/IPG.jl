using JuMP, JSON3

abstract type AbstractPlayer end

"A player in an IPG."
struct Player{Payoff<:AbstractPayoff} <: AbstractPlayer
    "Strategy space."
    X::Model
    """Strategy space variables.
    TODO: The idea is that the user will pass the player's variables to the constructor, so
    that it will be used to define the payoff function's arguments. See https://discourse.julialang.org/t/getting-jump-model-variables-and-containers-in-order-of-creation/127245/2
    """
    vars::Vector{<:Union{VariableRef, AbstractArray{VariableRef}}}
    "Payoff function."
    Π::Payoff
    "Player's index."
    p::Integer  # TODO: this could be Any, to allow for more general collections, e.g, string names
    # TODO: maybe I could just index everything relevant as a Dict{Player, T}?
    # TODO: I think with the new signature for the payoff functions the index is no longer needed
end

"Check whether an optimizer has already been set for player."
function has_optimizer(player::AbstractPlayer)
    return ~(backend(player.X).state == JuMP.MOIU.NO_OPTIMIZER)
end

"Define the optimizer for player."
function set_optimizer(player::AbstractPlayer, optimizer_factory)
    JuMP.set_optimizer(player.X, optimizer_factory)
end

"Compute `player`'s best response to the pure strategy profile `x`."
function best_response(player::Player{<:AbstractPayoff}, x::Vector{<:Vector{<:Real}})
    return best_response(player, DiscreteMixedStrategy.(x))
end

"Compute `player`'s best response to the mixed strategy profile `σp`."
function best_response(player::Player{<:AbstractPayoff}, σ::Vector{DiscreteMixedStrategy})
    # TODO: I think we could take only `σ_others` as argument
    xp = player.vars

    σ_others = others(σ, player.p)
    obj = expected_value(x_others -> payoff(player.Π, xp, x_others), σ_others)

    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.X, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.X, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.X)
    optimize!(player.X)

    return [value.(v) for v in xp]
end
function best_response(player::Player{<:AbstractBilateralPayoff}, σ::Vector{DiscreteMixedStrategy})
    xp = player.vars

    # TODO: No idea why this doesn't work
    # @objective(model, Max, sum([IPG.bilateral_payoff(Π, p, xp, k, σ[k]) for k in 1:m]))

    obj = AffExpr()
    for k in eachindex(σ)
        if k == player.p
            obj += IPG.bilateral_payoff(player.Π, xp)
        else
            obj += IPG.bilateral_payoff(player.Π, xp, σ[k], k)
        end
    end
    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.X, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.X, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.X)
    optimize!(player.X)

    return value.(xp)
end

"Solve the feasibility problem for a player, returning a feasible strategy."
function find_feasible_pure_strategy(player::AbstractPlayer)
    @objective(player.X, JuMP.MOI.FEASIBILITY_SENSE, 0)

    set_silent(player.X)
    optimize!(player.X)

    return [value.(v) for v in player.vars]
end

"Solve the feasibility problem of all players, returning a feasible profile."
function find_feasible_pure_profile(players::Vector{<:AbstractPlayer})
    return [find_feasible_pure_strategy(player) for player in players]
end

function save(player::Player{QuadraticPayoff}, filename::String)
    error("Not implemented")  # TODO: this has not been updated after the player.vars refactor
    # we need to ensure that the file is stored as a json, so we can add the payoff information
    JuMP.write_to_file(player.X, filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)

    # TODO: this could be refactored as a payoff JSON-serialization method
    # see https://quinnj.github.io/JSON3.jl/stable/#Struct-API
    mof_json = JSON3.read(read(filename, String))
    mof_json = copy(mof_json)  # JSON type is immutable

    mof_json[:IPG__player_index] = player.p
    mof_json[:IPG__payoff] = Dict(
        :cp => player.Π.cp,
        :Qp => player.Π.Qp,
        # JSON3 cannot store matrices, it stores them as a flat vector
        :Qp_shapes => [size(Qpk) for Qpk in player.Π.Qp],
        :p => player.Π.p,
    )

    open(filename, "w") do file
        JSON3.write(file, mof_json)
    end
end

function load(filename::String)::Player{QuadraticPayoff}
    error("Not implemented")  # TODO: this has not been updated after the player.vars refactor
    X = JuMP.read_from_file(filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)
    
    # see https://github.com/jump-dev/JuMP.jl/issues/3946
    set_start_value.(all_variables(X), start_value.(all_variables(X)))

    mof_json = JSON3.read(read(filename, String))

    player_index = mof_json[:IPG__player_index]
    payoff_data = mof_json[:IPG__payoff]
    cp = copy(payoff_data[:cp])
    flat_Qp = copy(payoff_data[:Qp])
    payoff_index = copy(payoff_data[:p])  # should be equal to player_index
    Qp_shapes = copy(payoff_data[:Qp_shapes])

    Qp = [reshape(flat_Qpk, Tuple(Qpk_shape)) for (flat_Qpk, Qpk_shape) in zip(flat_Qp, Qp_shapes)]

    return Player{QuadraticPayoff}(X, QuadraticPayoff(cp, Qp, payoff_index), player_index)
end
