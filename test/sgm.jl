include("utils.jl")


@testitem "Initialization" setup=[Utilities] begin
    players = get_example_two_player_game()
    for player in players
        IPG.set_optimizer(player, SCIP.Optimizer)
    end

    # remove start values from player 1
    for var in all_variables(players[1].X)
        set_start_value(var, nothing)
    end

    S_X = IPG.initialize_strategies_feasibility(players)

    @test Set(keys(S_X)) == Set(players)
    for player in players
        @test length(S_X[player]) == 1  # only one strategy should be initialized
    end
    @test S_X[players[2]] == [[10.0]]  # player 2 has a start value
    # check that player 1’s initialized strategy lies in its feasible region
    x1 = S_X[players[1]][1]
    @test length(x1) == length(all_variables(players[1]))
    @test all(x1 .>= 0)  # only constraint of player 1

    # remove start_values from player 2 as well
    for var in all_variables(players[2])
        set_start_value(var, nothing)
    end

    S_X = IPG.initialize_strategies_player_alone(players)

    @test Set(keys(S_X)) == Set(players)
    for player in players
        @test length(S_X[player]) == 1  # only one strategy should be initialized
        xp = S_X[player][1]
        @test length(xp) == length(all_variables(player))
        @test all(xp .== 0)  # known best response to 0
    end
end

@testitem "Deviation reaction" setup=[Utilities] begin
    players = get_example_two_player_game()
    for player in players
        IPG.set_optimizer(player, SCIP.Optimizer)
    end

    S_X = IPG.initialize_strategies(players)
    σ = Profile{DiscreteMixedStrategy}(player => S_X[player][1] for player in players)

    for player in players
        @test σ[player].supp == [[10.0]]  # has to be the start value
    end

    payoff_improvement, player, new_x_p = IPG.find_deviation(players, σ)

    previous_payoff = payoff(player, σ[player], others(σ, player))
    new_payoff = payoff(player, new_x_p, others(σ, player))

    @test payoff_improvement == new_payoff - previous_payoff
    @test payoff_improvement > 0.0  # there should be a deviation

    # there should be no deviation from an equilibrium
    σ_NE = Profile{DiscreteMixedStrategy}(player => [0.0] for player in players)
    payoff_improvement, player, new_x_p = IPG.find_deviation(players, σ_NE)
    @test payoff_improvement == 0.0
    @test isnothing(player)
    @test isnothing(new_x_p)
end

@testitem "Nonlinear deviation reaction" setup=[Utilities] begin
    # Example 5.3 from the IPG paper
    X1 = Model(SCIP.Optimizer)
    @variable(X1, x1, start=10.0)
    @constraint(X1, x1 >= 0)

    X2 = Model(SCIP.Optimizer)
    @variable(X2, x2, start=10.0)
    @constraint(X2, x2 >= 0)

    function player_payoff(x_self, x_other)
        return -x_self * x_self + x_self * x_other * x_other
    end

    players = [
        Player(X1, player_payoff(x1, x2)),
        Player(X2, player_payoff(x2, x1))
    ]
    for p in players
        @test p.Π isa NonlinearExpr
    end

    S_X = IPG.initialize_strategies(players)
    σ = Profile{DiscreteMixedStrategy}(player => S_X[player][1] for player in players)

    for player in players
        @test σ[player].supp == [[10.0]]  # has to be the start value
    end

    payoff_improvement, player, new_x_p = IPG.find_deviation(players, σ)

    previous_payoff = payoff(player, σ[player], others(σ, player))
    new_payoff = payoff(player, new_x_p, others(σ, player))

    @test payoff_improvement == new_payoff - previous_payoff
    @test payoff_improvement > 0.0  # there should be a deviation

    # there should be no deviation from an equilibrium
    σ_NE = Profile{DiscreteMixedStrategy}(player => [0.0] for player in players)
    payoff_improvement, player, new_x_p = IPG.find_deviation(players, σ_NE)

    @test payoff_improvement == 0.0
    @test isnothing(player)
    @test isnothing(new_x_p)
end

@testitem "simple two-player bilateral game (example 5.3)" setup=[Utilities] begin
    # guarantee reproducibility (always start with player 1)
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    players = get_example_two_player_game()

    Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=5, verbose=true);

    @test [σ[players[1]].supp for σ in Σ] ≈ [
        [[10.0]],
        [[10.0]],
        [[2.5]],
        [[2.5]],
        [[0.625]]
    ]
    @test [σ[players[2]].supp for σ in Σ] ≈ [
        [[10.0]],
        [[5.0]],
        [[5.0]],
        [[1.25]],
        [[1.25]]
    ]
    expected_improvements = [
        (players[2], 25.0),
        (players[1], 56.25),
        (players[2], 14.0625),
        (players[1], 3.515625),
        (players[2], 0.87890625)
    ]
    @test all(p_imp == p_expected for ((p_imp, _),(p_expected, _)) in zip(payoff_improvements, expected_improvements))
    @test all(imp ≈ expected_imp for ((_, imp),(_, expected_imp)) in zip(payoff_improvements, expected_improvements))
end

"SGM should work for any Nonlinear two-player game."
@testitem "Nonlinear example 5.3" setup=[Utilities] begin
    # guarantee reproducibility (always start with player 1)
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    # Example 5.3 from the IPG paper
    X1 = Model()
    @variable(X1, x1, start=10.0)
    @constraint(X1, x1 >= 0)

    X2 = Model()
    @variable(X2, x2, start=10.0)
    @constraint(X2, x2 >= 0)

    function player_payoff(x_self, x_other)
        return -x_self * x_self + x_self * x_other
    end

    players = [
        Player(X1, NonlinearExpr(:+, Any[player_payoff(x1, x2)])),
        Player(X2, NonlinearExpr(:+, Any[player_payoff(x2, x1)])),
    ]

    Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=5, verbose=true);

    @test [σ[players[1]].supp for σ in Σ] ≈ [
        [[10.0]],
        [[10.0]],
        [[2.5]],
        [[2.5]],
        [[0.625]]
    ]
    @test [σ[players[2]].supp for σ in Σ] ≈ [
        [[10.0]],
        [[5.0]],
        [[5.0]],
        [[1.25]],
        [[1.25]]
    ]
    expected_improvements = [
        (players[2], 25.0),
        (players[1], 56.25),
        (players[2], 14.0625),
        (players[1], 3.515625),
        (players[2], 0.87890625)
    ]
    @test all(p_imp == p_expected for ((p_imp, _),(p_expected, _)) in zip(payoff_improvements, expected_improvements))
    @test all(imp ≈ expected_imp for ((_, imp),(_, expected_imp)) in zip(payoff_improvements, expected_improvements))
end
