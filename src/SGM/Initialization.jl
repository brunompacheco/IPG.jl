
function initialize_strategies_feasibility(players::Vector{<:AbstractPlayer})
    S_X = [[] for _ in players]
    for player in players
        xp_init = []
        run_feasibility = false
        for v in player.vars
            if v isa VariableRef
                v_init = start_value(v)
            elseif v isa AbstractArray{VariableRef}
                v_init = start_value.(v)
            end

            if any(isnothing.(v_init))
                run_feasibility = true
                break
            else
                push!(xp_init, v_init)
            end
        end

        if run_feasibility
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            xp_init = find_feasible_pure_strategy(player)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

function initialize_strategies_player_alone(players::Vector{<:AbstractPlayer})
    S_X = [Vector{Vector{Float64}}() for _ in players]

    # mixed profile that simulates players being alone (all others play 0)
    function zeros_like_vars(player::Player)
        z = []
        for v in player.vars
            if v isa VariableRef
                push!(z, 0.0)
            elseif v isa AbstractArray{VariableRef}
                push!(z, zeros(size(v)))
            else
                error("Variable type not supported: ", typeof(v))
            end
        end
        return z
    end
    σ_dummy = [DiscreteMixedStrategy(zeros_like_vars(player)) for player in players]

    for player in players
        xp_init = []
        run_feasibility = false
        for v in player.vars
            if v isa VariableRef
                v_init = start_value(v)
            elseif v isa AbstractArray{VariableRef}
                v_init = start_value.(v)
            end

            if any(isnothing.(v_init))
                run_feasibility = true
                break
            else
                push!(xp_init, v_init)
            end
        end

        if run_feasibility
            # TODO: if `initial_sol` is just a partial solution, I could fix its values
            # before solving the feasibility problem.
            xp_init = best_response(player, σ_dummy)
        end

        push!(S_X[player.p], xp_init)
    end

    return S_X
end

initialize_strategies = initialize_strategies_feasibility
