using JuMP, JSON3

const VarToParamDict = Dict{VariableRef,VariableRef}

"A player in an IPG."
mutable struct Player
    "Strategy space."
    X::Model
    # TODO: using value(...) to manipulate expressions does not work for NonlinearExpr, see https://github.com/jump-dev/JuMP.jl/issues/4044 for an appropriate solution (another huge refactor))
    "Payoff expression."
    Π::AbstractJuMPScalar
    _param_dict::VarToParamDict
end
Player() = Player(Model(), AffExpr(NaN), VarToParamDict())
Player(X::Model) = Player(X, AffExpr(NaN), VarToParamDict())
function Player(X::Model, Π::AbstractJuMPScalar) 
    player = Player(X)
    set_payoff!(player, Π)
    return player
end
export Player

JuMP.all_variables(p::Player) = filter(v -> ~is_parameter(v), all_variables(p.X))

"Maps external variables to internal parameters. Creates a new parameter if it does not exist."
function _maybe_create_parameter_for_external_var(player::Player, var::VariableRef)::VariableRef
    var ∈ all_variables(player.X) && return var

    if ~haskey(player._param_dict, var)
        # create anonymous parameter with the same name as the variable
        param = @variable(player.X, base_name=name(var), set=Parameter(NaN))

        player._param_dict[var] = param
    end

    return player._param_dict[var]
end

function set_payoff!(player::Player, payoff::AbstractJuMPScalar)
    function _recursive_internalize_expr(expr::AbstractJuMPScalar)::AbstractJuMPScalar
        if expr isa VariableRef
            return _maybe_create_parameter_for_external_var(player, expr)
        elseif expr isa AffExpr
            internal_terms = typeof(expr.terms)(
                _maybe_create_parameter_for_external_var(player, var) => coeff
                for (var, coeff) in expr.terms
            )
            return AffExpr(expr.constant, )
        elseif expr isa QuadExpr
            internal_terms = typeof(expr.terms)(
                UnorderedPair{VariableRef}(
                    _maybe_create_parameter_for_external_var(player, vars.a),
                    _maybe_create_parameter_for_external_var(player, vars.b)
                ) => coeff
                for (vars, coeff) in expr.terms
            )
            return QuadExpr(_recursive_internalize_expr(expr.aff), internal_terms)
        elseif expr isa NonlinearExpr
            return NonlinearExpr(expr.head, Vector{Any}(map(_recursive_internalize_expr, expr.args)))
        else
            return expr
        end
    end

    player.Π = _recursive_internalize_expr(payoff)
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

    return value.(all_variables(player))
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
