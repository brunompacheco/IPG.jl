using JuMP, JSON3

abstract type AbstractPlayer end

"A player in an IPG."
struct Player{Payoff<:AbstractPayoff} <: AbstractPlayer
    "Strategy space."
    Xp::Model
    "Payoff function."
    Πp::Payoff
    "Player's index."
    p::Integer  # TODO: this could be Any, to allow for more general collections, e.g, string names
    # TODO: maybe I could just index everything relevant as a Dict{Player, T}?
    # TODO: I think with the new signature for the payoff functions the index is no longer needed
end
"Initialize player with empty strategy space."
function Player(Πp::Payoff, p::Integer) where Payoff <: AbstractPayoff
    return Player{Payoff}(Model(), Πp, p)
end

# TODO: clarify in the README how the variables are passed from the model to the payoff function.
"""
Get all player registered variables (`xp`, for the payoff function).

This function differs from `JuMP.all_variables(player.Xp)` in that it returns the variables
in their container. It uses the model's object dictionary, so we will only get registered
variables.
"""
function variables(player::AbstractPlayer)
    # TODO: I'm not quite sure which is best, but another way of doing this would be to get
    # the return from `all_variables` and whenever we get a name that is registered, we
    # replace it (and all its occurences) by the object in the dictionary. I don't know
    # which is best for the user though.
    vars = [
        v for v in values(object_dictionary(player.Xp))
        if (v isa VariableRef) | (v isa AbstractArray{VariableRef})
    ]

    sort!(vars, by=v -> findfirst(u -> u == (v isa VariableRef ? v : first(v)), all_variables(player.Xp)))
end


"Check whether an optimizer has already been set for player."
function has_optimizer(player::AbstractPlayer)
    return ~(backend(player.Xp).state == JuMP.MOIU.NO_OPTIMIZER)
end

"Define the optimizer for player."
function set_optimizer(player::AbstractPlayer, optimizer_factory)
    JuMP.set_optimizer(player.Xp, optimizer_factory)
end

"Compute `player`'s best response to the pure strategy profile `x`."
function best_response(player::Player{<:AbstractPayoff}, x::Vector{<:PureStrategy})
    return best_response(player, DiscreteMixedStrategy.(x))
end

"Compute `player`'s best response to the mixed strategy profile `σp`."
function best_response(player::Player{<:AbstractPayoff}, σ::Vector{<:DiscreteMixedStrategy})
    # TODO: I think we could take only `σ_others` as argument
    xp = variables(player)

    σ_others = others(σ, player.p)
    obj = expected_value(x_others -> payoff(player.Πp, xp, x_others), σ_others)

    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.Xp, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.Xp, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(xp)
end
function best_response(player::Player{<:AbstractBilateralPayoff}, σ::Vector{<:DiscreteMixedStrategy})
    xp = variables(player)

    # TODO: No idea why this doesn't work
    # @objective(model, Max, sum([IPG.bilateral_payoff(Πp, p, xp, k, σ[k]) for k in 1:m]))

    obj = AffExpr()
    for k in eachindex(σ)
        if k == player.p
            obj += IPG.bilateral_payoff(player.Πp, xp)
        else
            obj += IPG.bilateral_payoff(player.Πp, xp, σ[k], k)
        end
    end
    # I don't know why, but it was raising an error without changing the sense to feasibility first
    set_objective_sense(player.Xp, JuMP.MOI.FEASIBILITY_SENSE)
    @objective(player.Xp, JuMP.MOI.MAX_SENSE, obj)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(xp)
end

"Solve the feasibility problem for a player, returning a feasible strategy."
function find_feasible_pure_strategy(player::AbstractPlayer)
    @objective(player.Xp, JuMP.MOI.FEASIBILITY_SENSE, 0)

    set_silent(player.Xp)
    optimize!(player.Xp)

    return value.(variables(player))
end

"Solve the feasibility problem of all players, returning a feasible profile."
function find_feasible_pure_profile(players::Vector{<:AbstractPlayer})
    return [find_feasible_pure_strategy(player) for player in players]
end

function save(player::Player{QuadraticPayoff}, filename::String)
    # we need to ensure that the file is stored as a json, so we can add the payoff information
    JuMP.write_to_file(player.Xp, filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)

    # TODO: this could be refactored as a payoff JSON-serialization method
    # see https://quinnj.github.io/JSON3.jl/stable/#Struct-API
    mof_json = JSON3.read(read(filename, String))
    mof_json = copy(mof_json)  # JSON type is immutable

    mof_json[:IPG__player_index] = player.p
    mof_json[:IPG__payoff] = Dict(
        :cp => player.Πp.cp,
        :Qp => player.Πp.Qp,
        # JSON3 cannot store matrices, it stores them as a flat vector
        :Qp_shapes => [size(Qpk) for Qpk in player.Πp.Qp],
        :p => player.Πp.p,
    )

    open(filename, "w") do file
        JSON3.write(file, mof_json)
    end
end

function load(filename::String)::Player{QuadraticPayoff}
    Xp = JuMP.read_from_file(filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)
    
    # see https://github.com/jump-dev/JuMP.jl/issues/3946
    set_start_value.(all_variables(Xp), start_value.(all_variables(Xp)))

    mof_json = JSON3.read(read(filename, String))

    player_index = mof_json[:IPG__player_index]
    payoff_data = mof_json[:IPG__payoff]
    cp = copy(payoff_data[:cp])
    flat_Qp = copy(payoff_data[:Qp])
    payoff_index = copy(payoff_data[:p])  # should be equal to player_index
    Qp_shapes = copy(payoff_data[:Qp_shapes])

    Qp = [reshape(flat_Qpk, Tuple(Qpk_shape)) for (flat_Qpk, Qpk_shape) in zip(flat_Qp, Qp_shapes)]

    return Player{QuadraticPayoff}(Xp, QuadraticPayoff(cp, Qp, payoff_index), player_index)
end
