include("utils.jl")

@testitem "Player construction" begin
    using JuMP

    # TODO: extend this test to players with multiple variables
    # Example 5.3 from the IPG paper
    X1 = Model()
    @variable(X1, x1, start=10.0)
    @constraint(X1, x1 >= 0)

    X2 = Model()
    @variable(X2, x2, start=10.0)
    @constraint(X2, x2 >= 0)

    players = [Player(X1), Player(X2)]

    for p in players
        # vars should match, as we haven't added any parameter, i.e, one player is "unaware" of the other
        @test all_variables(p) == all_variables(p.X)
    end

    set_payoff!(players[1], -x1 * x1 + x1 * x2)
    set_payoff!(players[2], x2 / (x1*x1))  # just some nonlinear function that is actually linear for the player
    @assert players[2].Π isa NonlinearExpr

    for p in players
        @test owner_model(p.Π) === p.X
    end

    @test collect(keys(players[1]._param_dict)) == all_variables(players[2])
    @test collect(keys(players[2]._param_dict)) == all_variables(players[1])

    x1_bar = [20.0]
    x2_bar = [20.0]
    v1_bar = Assignment(players[1], x1_bar)
    v2_bar = Assignment(players[2], x2_bar)

    payoff_res = payoff(players[1], x1_bar, Profile{PureStrategy}(players[2] => x2_bar))
    best_response_payoff_p1 = IPG.replace_in_payoff(players[1], v2_bar)
    simplified_res = value(v -> v1_bar[v], best_response_payoff_p1)

    @test simplified_res == payoff_res

    payoff_res = payoff(players[2], x2_bar, Profile{PureStrategy}(players[1] => x1_bar))
    best_response_payoff_p2 = IPG.replace_in_payoff(players[2], v1_bar)
    println(typeof(best_response_payoff_p2))
    @test best_response_payoff_p2 isa AffExpr
    simplified_res = value(v -> v2_bar[v], best_response_payoff_p2)

    @test simplified_res == payoff_res
end

@testitem "Mixed strategies in payoffs" setup=[Utilities] begin
    X = Model(); @variable(X, x[1:2])
    Y = Model(); @variable(Y, y[1:2])

    player1 = Player(X, x'* y)
    player2 = Player(Y, y[1] * sqrt(x[1] / x[2]))

    σ_y = DiscreteMixedStrategy([0.2, 0.8], [[10., 0.], [0., 10.]])
    obj_expr = expected_value(
        y_bar -> IPG.replace_in_payoff(player1, Assignment(y_bar)),
        Dict(player2 => σ_y)
    )
    @test obj_expr == 2*x[1] + 8*x[2]

    σ_x = DiscreteMixedStrategy([0.7, 0.3], [[4., 1.], [1., 4.]])
    obj_expr = expected_value(
        x_bar -> IPG.replace_in_payoff(player2, Assignment(x_bar)),
        Dict(player1 => σ_x)
    )
    @test obj_expr == (0.7*2 + 0.3*0.5) * y[1]
end

@testitem "Best Response pure profile" setup=[Utilities] begin
    X = Model(SCIP.Optimizer)
    @variable(X, x[1:2])
    @constraint(X, 1 .<= 2 .* x .+ 1 .<= 3)  # dummy unit cube

    player1 = Player(X, x[1] * x[2])

    @objective(X, Max, x[1] * x[2])

    set_silent(X)
    optimize!(X)

    x_opt = value.(x)

    Y = Model()
    @variable(Y, y[1:2])
    @constraint(Y, 1 .<= 2 .* y .+ 1 .<= 3)  # dummy unit cube

    player2 = Player(Y)

    # dummy profile
    pure_profile = Dict(player2 => [1.0, 1.0])
    best_x1 = IPG.best_response(player1, pure_profile)
    @test length(best_x1) == 2
    @test best_x1 == x_opt == [1.0, 1.0]  # the best response is always x = (1,1)
end

@testitem "Best response mixed profile" setup=[Utilities] begin
    X = Model(SCIP.Optimizer)
    @variable(X, x[1:2], start=0.5)
    @constraint(X, 1 .<= 2 .* x .+ 1 .<= 3)  # dummy unit cube
    @constraint(X, x[1] + x[2] == 1)  # constraint to make it non-trivial

    Y = Model()
    @variable(Y, y[1:2])
    @constraint(Y, 1 .<= 2 .* y .+ 1 .<= 3)  # dummy unit cube

    player1 = Player(X, x'* y)
    player2 = Player(Y)

    σ_player2_1 = DiscreteMixedStrategy([0.5, 0.5], [[1., 0.], [0., 1.]])
    @test IPG.best_response(player1, Dict(player2 => σ_player2_1)) == [0.5, 0.5]  # start value is optimal

    σ_player2_3 = DiscreteMixedStrategy([0.4, 0.6], [[1., 0.], [0., 1.]])
    @test IPG.best_response(player1, Dict(player2 => σ_player2_3)) == [0.0, 1.0]

    σ_player2_2 = DiscreteMixedStrategy([0.6, 0.4], [[1., 0.], [0., 1.]])
    @test IPG.best_response(player1, Dict(player2 => σ_player2_2)) == [1.0, 0.0]
end

@testitem "DiscreteMixedStrategy" begin
    # TODO: really annoying that I have to pass the support (PureStrategy) as floats. this happens because the vector is not automatically promoted when there is another argument in the method.
    σp = DiscreteMixedStrategy([0.5, 0.5], [[1., 0.], [0., 1.]])
    @test length(σp.probs) == 2
    @test size(σp.supp) == (2,)
    @test size(σp.supp[1]) == (2,)
    @test size(σp.supp[2]) == (2,)
    @test expected_value(identity, σp) == [0.5, 0.5]
    @test expected_value(sum, σp) == 1.0

    xp = [1., 2., 3.]
    σp = convert(DiscreteMixedStrategy, xp)
    @test expected_value(identity, σp) == xp

    σa = DiscreteMixedStrategy([0.5, 0.5], [[1., 0.], [0., 1.]])
    σb = DiscreteMixedStrategy([0.5, 0.5], [[1., 0.], [0., 1.]])
    @test σa == σb

    σc = DiscreteMixedStrategy([0.5 + eps(), 0.5 - eps()], [[1., 0.], [0., 1.]])
    @test σa != σc
    @test σa ≈ σc
end

@testitem "Assignments" setup=[Utilities] begin
    players = get_example_two_player_game()
    x1_bar = [20.0]
    x2_bar = [20.0]
    v1_bar = Assignment(players[1], x1_bar)
    v2_bar = Assignment(players[2], x2_bar)

    payoff_res = payoff(players[1], x1_bar, Dict(players[2] => x2_bar))

    v2_bar_for_p1 = IPG._internalize_assignment(players[1], v2_bar)
    best_response_payoff_p1 = IPG.replace(players[1].Π, v2_bar_for_p1)
    simplified_res = value(v -> v1_bar[v], best_response_payoff_p1)

    @test simplified_res == payoff_res

    x1 = all_variables(players[1].X)[1]
    x2 = all_variables(players[2].X)[1]
    expr = (x1*x2) / (2*x1)
    @assert expr isa NonlinearExpr

    replaced_expr = IPG.replace(expr, IPG.AssignmentDict(x1 => 1.0))

    @test owner_model(replaced_expr) === players[2].X
    @test replaced_expr isa AffExpr
end

@testitem "Finding feasible strategies" setup=[Utilities] begin
    X = Model()
    @variable(X, x[1:2])
    @constraint(X, 1 .<= 2 .* x .+ 1 .<= 3)  # dummy unit cube

    player1 = Player(X, x[1] * x[2])
    IPG.set_optimizer(player1, SCIP.Optimizer)

    Y = Model()
    @variable(Y, y[1:2])
    @constraint(Y, 1 .<= 2 .* y .+ 1 .<= 3)  # dummy unit cube

    player2 = Player(Y, y[1] * y[2])
    IPG.set_optimizer(player2, SCIP.Optimizer)

    x1 = IPG.find_feasible_pure_strategy(player1)

    @test length(x1) == 2
    @test all(x1 .>= 0)
    @test all(x1 .<= 1.0)

    x_all = IPG.find_feasible_pure_profile([player1, player2])

    y2 = x_all[player2]

    @test length(y2) == 2
    @test all(y2 .>= 0)
    @test all(y2 .<= 1.0)
end

@testitem "Assignments" setup=[Utilities] begin
    players = get_example_two_player_game()

    assignment_p2 = Assignment(players[2], start_value.(all_variables(players[2])))
    p2_assignment_refs = collect(keys(assignment_p2))
    @test p2_assignment_refs ⊈ all_variables(players[1].X)
    @test p2_assignment_refs == all_variables(players[2])

    internalized_assignment_p2_p2 = IPG._internalize_assignment(players[2], assignment_p2)
    @test internalized_assignment_p2_p2 == assignment_p2

    internalized_assignment_p2_p1 = IPG._internalize_assignment(players[1], assignment_p2)
    p2_assignment_refs_internalized_p1 = collect(keys(internalized_assignment_p2_p1))
    @test p2_assignment_refs_internalized_p1 ⊈ all_variables(players[2].X)
    @test p2_assignment_refs_internalized_p1 ⊈ all_variables(players[1])
    @test p2_assignment_refs_internalized_p1 ⊆ all_variables(players[1].X)
end

@testitem "Payoff" setup=[Utilities] begin
    players = get_example_two_player_game()

    p1_x_bar = p2_x_bar = [100.0*rand()+1.0]
    x_others = Profile{PureStrategy}(players[2] => p2_x_bar)

    p1_payoff_fun = IPG.get_payoff_map(players[1], x_others)
    @test p1_payoff_fun(p1_x_bar) == payoff(players[1], p1_x_bar, x_others) == 0.0

    p2_x_bar = [0.0]
    x_others = Profile{PureStrategy}(players[2] => p2_x_bar)
    @test payoff(players[1], p1_x_bar, x_others) < 0.0

    p2_x_bar = p1_x_bar / 2
    x_others = Profile{PureStrategy}(players[1] => p1_x_bar)
    @test payoff(players[2], p2_x_bar, x_others) > 0.0

    x_others = Profile{PureStrategy}(players[2] => p2_x_bar)
    p2_σ = DiscreteMixedStrategy([0.3, 0.7], [p2_x_bar, p2_x_bar])
    σ_others = Profile{DiscreteMixedStrategy}(players[2] => p2_σ)
    payoff_mixed = payoff(players[1], p1_x_bar, σ_others)
    payoff_pure = payoff(players[1], p1_x_bar, x_others)
    @test payoff_mixed ≈ payoff_pure
end
