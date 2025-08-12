using IPG
using TestItems

@testsnippet Utilities begin
    using LinearAlgebra, JuMP, SCIP

    function generate_random_instance(n::Int, m::Int, lower_bnd::Int, upper_bnd::Int; i_type="H")
        factor = i_type == "H" ? 0.1 : 0.01
        RQ = 5

        # Generate positive semidefinite matrix M
        M = zeros(Float64, (n*m, n*m))
        while ~isposdef(M)
            M = rand(Float64, (n*m, n*m))
            M = (M .* 2 .- 1) .* RQ  # scaling
            M = M * M'
        end

        M_max = maximum(M)
        for i in 1:n
            for j in ((i-1) * m + 1):(i * m)
                for k in (i * m + 1):(size(M, 2))
                    vjk = (rand() * 2 - 1) * factor * M_max
                    vjk = round(vjk, digits=1)
                    M[j, k] += vjk
                    M[k, j] -= vjk
                end
            end
        end

        # build strategy spaces
        X = [Model() for _ in 1:n]
        for p in 1:n
            @variable(X[p], [1:m], Int, base_name="x_$p", lower_bound=lower_bnd, upper_bound=upper_bnd)
        end
        x = [[variable_by_name(X[p], "x_$p[$i]") for i in 1:m] for p in 1:n]

        # build players
        players = Vector{Player}()
        for p in 1:n
            # build payoff
            Qp = Vector{Matrix{Float64}}()
            for k in 1:n
                push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
            end

            cp = rand(-RQ:RQ, m)
            
            Πp = cp'*x[p]
            Πp += 0.5 * x[p]' * Qp[p] * x[p]
            Πp += sum(x[k]' * Qp[k] * x[p] for k in 1:n if k != p)

            # build strategy space
            push!(players, Player(X[p], Πp))
        end

        return players
    end

    function get_example_two_player_game()
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

        return [
            Player(X1, player_payoff(x1, x2)),
            Player(X2, player_payoff(x2, x1))
        ]
    end
end

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

@testitem "Polymatrix computation" setup=[Utilities] begin
    players = get_example_two_player_game()
    for player in players
        IPG.set_optimizer(player, SCIP.Optimizer)
    end

    # give some options so that we can test the polymatrix
    S_X = Dict(players[1] => [[10.0],[5.0]], players[2]=> [[10.0],[5.0]])

    polymatrix = IPG.get_polymatrix_bilateral(players, S_X)

    for p in players
        for x_pure in S_X[p]
            @test IPG.compute_self_payoff(p, x_pure) == - x_pure[1]^2
        end
    end

    p1, p2 = players
    @test IPG.compute_bilateral_payoff(p1, S_X[p1][1], p2, S_X[p2][1]) == IPG.compute_bilateral_payoff(p2, S_X[p2][1], p1, S_X[p1][1]) == 10*10

    @test polymatrix[players[1], players[1]] == polymatrix[players[2], players[2]] == zeros(2, 2)
    @test polymatrix[players[1], players[2]] == polymatrix[players[2], players[1]]
    @test polymatrix[players[1], players[2]]== [ 0.0 -50.0; 25.0 0.0 ]

    two_player_polymatrix = IPG.get_polymatrix_twoplayers(players[1], players[2], S_X)

    @test two_player_polymatrix == polymatrix

    incremental_S_X = IPG.initialize_strategies(players)  # initialized from start values
    sampled_game = IPG.PolymatrixSampledGame(players, incremental_S_X)
    IPG.add_new_strategy!(sampled_game, players[1], [5.0])
    IPG.add_new_strategy!(sampled_game, players[2], [5.0])

    @test sampled_game.polymatrix == polymatrix
end

@testitem "Solving polymatrix game" setup=[Utilities] begin
    players = get_example_two_player_game()

    S_X = Dict(players[1] => [[10.0],[5.0]], players[2]=> [[1.0],[5.0]])

    sampled_game = IPG.PolymatrixSampledGame(players, S_X)

    σ_PNS = IPG.solve_PNS(sampled_game, SCIP.Optimizer)
    σ_Sandholm = IPG.solve_Sandholm1(sampled_game, SCIP.Optimizer)

    @test expected_value(identity, σ_PNS[players[1]]) == expected_value(identity, σ_Sandholm[players[1]]) == [5.0]
    @test expected_value(identity, σ_PNS[players[2]]) == expected_value(identity, σ_Sandholm[players[2]]) == [1.0]
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

@testitem "README Example Test" begin
    using IPG, SCIP

    # this is necessary for reproducibility, but doesn't affect the user experience
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    P1 = Player()
    P2 = Player()

    @variable(P1.X, x1, start=10)

    @constraint(P1.X, x1 >= 0)

    @variable(P2.X, x2, start=10)

    @constraint(P2.X, x2 >= 0)

    set_payoff!(P1, -x1*x1 + x1*x2)
    @test string(P1.Π) == string(-x1*x1 + x1*x2)

    set_payoff!(P2, -x2*x2 + x1*x2)
    @test string(P2.Π) == string(-x2*x2 + x1*x2)

    Σ, payoff_improvements = SGM([P1, P2], SCIP.Optimizer, max_iter=5)

    # Verify the final strategies match the expected values
    @test Σ[end][P1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
    @test Σ[end][P2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
end

# The following tests on the examples/ should mostly guarantee that they run without errors.

@testitem "Example 5.3" begin
    include("../examples/example_5_3.jl")

    # TODO: this is currently our only (easy) way to check that an equilibrium was found.
    # And I'm not even sure that this is 100% reliable.
    @test length(payoff_improvements) == length(Σ) - 1
end

@testitem "Example CFLD" begin
    include("../examples/cfld.jl")

    @test length(payoff_improvements) <= length(Σ)
end

@testitem "Example qIPG" begin
    include("../examples/quad_game.jl")

    @test length(payoff_improvements) <= length(Σ)
end
