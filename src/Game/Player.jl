using JuMP, JSON3

"A player in an IPG."
mutable struct Player
    "Strategy space."
    X::Model
    # TODO: using value(...) to manipulate expressions does not work for NonlinearExpr, see https://github.com/jump-dev/JuMP.jl/issues/4044 for an appropriate solution (another huge refactor))
    "Payoff expression."
    Π::AbstractJuMPScalar
end
function Player()
    return Player(Model(), AffExpr(0))
end
export Player

function set_payoff!(player::Player, payoff::AbstractJuMPScalar)
    player.Π = payoff
end
function set_payoff!(player::Player, payoff::Real)
    player.Π = AffExpr(payoff)
end
export set_payoff!

"Check whether an optimizer has already been set for player."
function has_optimizer(player::Player)
    return ~(backend(player.X).state == JuMP.MOIU.NO_OPTIMIZER)
end

"Define the optimizer for player."
function set_optimizer(player::Player, optimizer_factory)
    JuMP.set_optimizer(player.X, optimizer_factory)
end
export set_optimizer

"Solve the feasibility problem for a player, returning a feasible strategy."
function find_feasible_pure_strategy(player::Player)::PureStrategy
    @objective(player.X, JuMP.MOI.FEASIBILITY_SENSE, 0)

    set_silent(player.X)
    optimize!(player.X)

    return value.(all_variables(player.X))
end

"Solve the feasibility problem of all players, returning a feasible profile."
function find_feasible_pure_profile(players::Vector{Player})::Profile{PureStrategy}
    return Profile{PureStrategy}(player => find_feasible_pure_strategy(player) for player in players)
end

# TODO: serialization is discontinued (for now)
# function save(player::Player{<:AbstractJuMPScalar}, filename::String)
#     # we need to ensure that the file is stored as a json, so we can add the payoff information
#     JuMP.write_to_file(player.X, filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)

#     # TODO: this could be refactored as a payoff JSON-serialization method
#     # see https://quinnj.github.io/JSON3.jl/stable/#Struct-API
#     mof_json = JSON3.read(read(filename, String))
#     mof_json = copy(mof_json)  # JSON type is immutable

#     mof_json[:IPG__player_index] = player.p
#     mof_json[:IPG__payoff] = Dict(
#         :cp => player.Π.cp,
#         :Qp => player.Π.Qp,
#         # JSON3 cannot store matrices, it stores them as a flat vector
#         :Qp_shapes => [size(Qpk) for Qpk in player.Π.Qp],
#         :p => player.Π.p,
#     )

#     open(filename, "w") do file
#         JSON3.write(file, mof_json)
#     end
# end

# function load(filename::String)::Player{QuadraticPayoff}
#     Xp = JuMP.read_from_file(filename; format = JuMP.MOI.FileFormats.FORMAT_MOF)
    
#     # see https://github.com/jump-dev/JuMP.jl/issues/3946
#     set_start_value.(all_variables(Xp), start_value.(all_variables(Xp)))

#     mof_json = JSON3.read(read(filename, String))

#     player_index = mof_json[:IPG__player_index]
#     payoff_data = mof_json[:IPG__payoff]
#     cp = copy(payoff_data[:cp])
#     flat_Qp = copy(payoff_data[:Qp])
#     payoff_index = copy(payoff_data[:p])  # should be equal to player_index
#     Qp_shapes = copy(payoff_data[:Qp_shapes])

#     Qp = [reshape(flat_Qpk, Tuple(Qpk_shape)) for (flat_Qpk, Qpk_shape) in zip(flat_Qp, Qp_shapes)]

#     return Player{QuadraticPayoff}(Xp, QuadraticPayoff(cp, Qp, payoff_index), player_index)
# end
