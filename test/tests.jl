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

    pure_profile = Dict(player2 => y2)
    best_x1 = IPG.best_response(player1, pure_profile)
    @test length(best_x1) == 2
    @test best_x1 == [1.0, 1.0]  # the best response is always x = (1,1)
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
    @test length(x1) == length(all_variables(players[1].X))
    @test all(x1 .>= 0)  # only constraint of player 1

    # remove start_values from player 2 as well
    for var in all_variables(players[2].X)
        set_start_value(var, nothing)
    end

    S_X = IPG.initialize_strategies_player_alone(players)

    @test Set(keys(S_X)) == Set(players)
    for player in players
        @test length(S_X[player]) == 1  # only one strategy should be initialized
        xp = S_X[player][1]
        @test length(xp) == length(all_variables(player.X))
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
    players = [Player(p.X, convert(JuMP.GenericNonlinearExpr, p.Π)) for p in get_example_two_player_game()]
    for player in players
        IPG.set_optimizer(player, SCIP.Optimizer)
    end

    S_X = IPG.initialize_strategies(players)
    σ = Profile{DiscreteMixedStrategy}(player => S_X[player][1] for player in players)

    for player in players
        @test σ[player].supp == [[10.0]]  # has to be the start value
    end

    # TODO: there is a bug in JuMP when evaluating nonlinear expressions with variable
    # references. see https://github.com/jump-dev/JuMP.jl/issues/4044. until the issue is
    # fixed, this will likely not work.
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

    @test polymatrix[players[1], players[1]] == polymatrix[players[2], players[2]] == zeros(2, 2)
    @test polymatrix[players[1], players[2]] == polymatrix[players[2], players[1]] == [
        0.0 -50.0;
        25.0 0.0
    ]

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

@testitem "Example 5.3" setup=[Utilities] begin
    # guarantee reproducibility (always start with player 1)
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    players = get_example_two_player_game()

    Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5, verbose=true);

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
        Player(X1, convert(JuMP.GenericNonlinearExpr, player_payoff(x1, x2))),
        Player(X2, convert(JuMP.GenericNonlinearExpr, player_payoff(x2, x1)))
    ]

    Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5, verbose=true);

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

    P1 = Player()
    P2 = Player()

    @variable(P1.X, x1, start=10)

    @constraint(P1.X, x1 >= 0)

    @variable(P2.X, x2, start=10)

    @constraint(P2.X, x2 >= 0)

    set_payoff!(P1, -x1*x1 + x1*x2)
    @test P1.Π == -x1*x1 + x1*x2

    set_payoff!(P2, -x2*x2 + x1*x2)
    @test P2.Π == -x2*x2 + x1*x2

    Σ, payoff_improvements = IPG.SGM([P1, P2], SCIP.Optimizer, max_iter=5)

    # Verify the final strategies match the expected values
    @test Σ[end][P1] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
    @test Σ[end][P2] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
end

# TODO: this is also waiting for the issue on NonlinearExpr to be fixed (aka, the new refactor)
# @testitem "README Example Two-player" begin
#     player_payoff(xp, x_others) = -(xp[1] * xp[1]) + xp[1] * prod(x_others[:][1])

#     players = [
#         Player(BlackBoxPayoff(player_payoff), 1),
#         Player(BlackBoxPayoff(player_payoff), 2)
#     ];

#     @variable(players[1].X, x1, start=10); @constraint(players[1].X, x1 >= 0);

#     @variable(players[2].X, x2, start=10); @constraint(players[2].X, x2 >= 0);

#     Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5);

#     # Verify the final strategies match the expected values
#     @test Σ[end][1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
#     @test Σ[end][2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
# end

# @testitem "Player serialization" begin
#     X1 = Model()
#     @variable(X1, x1, start=10.0)
#     @constraint(X1, x1 >= 0)

#     player = Player(X1, QuadraticPayoff(0, [2, 1], 1), 1)

#     filename = "test_player.json"
#     IPG.save(player, filename)
#     loaded_player = IPG.load(filename)

#     @test loaded_player.p == player.p
#     @test loaded_player.Π.cp == player.Π.cp
#     @test loaded_player.Π.Qp == player.Π.Qp

#     # test strategy space
#     X1 = player.X
#     loaded_X1 = loaded_player.X

#     set_optimizer(X1, SCIP.Optimizer)
#     set_optimizer(loaded_X1, SCIP.Optimizer)

#     set_silent(X1)
#     optimize!(X1)

#     set_silent(loaded_X1)
#     optimize!(loaded_X1)

#     @test value.(all_variables(X1)) == value.(all_variables(loaded_X1))
#     @test objective_value(X1) == objective_value(loaded_X1)

#     # cleanup
#     rm(filename)
# end
