using IPG
using TestItems

@testsnippet Utilities begin
    using LinearAlgebra, JuMP, SCIP

    function generate_random_instance(n::Int, m::Int, lower_bound::Int, upper_bound::Int; i_type="H")
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

        # build players
        players = Vector{Player{QuadraticPayoff}}()
        for p in 1:n
            # build payoff
            Qp = Vector{Matrix{Float64}}()
            for k in 1:n
                push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
            end

            cp = rand(-RQ:RQ, m)
            Πp = QuadraticPayoff(cp, Qp, p)

            # build strategy space
            Xp = Model()
            @variable(Xp, lower_bound <= x[1:m] <= upper_bound, Int)

            push!(players, Player(Xp, Πp, p))
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

@testitem "Two-player game" begin
    IPG.initialize_strategies = IPG.initialize_strategies_player_alone

    bilateral_players = generate_random_instance(2, 2, -5, 5)
    for player in bilateral_players
        for variable in all_variables(player.X)
            set_start_value(variable, 1.0)
        end
    end

    # "black-box" function for blackbox payoff players
    function get_blackbox_function(p)
        return (xp, x_others) -> payoff(bilateral_players[p].Π, xp, x_others)
    end

    blackbox_players = [
        Player(copy(player.X), BlackBoxPayoff(get_blackbox_function(player.p)), player.p)
        for player in bilateral_players
    ]

    Σ_bilateral, poff_imp_bilateral = IPG.SGM(bilateral_players, SCIP.Optimizer, max_iter=10)
    Σ_blackbox, poff_imp_blackbox = IPG.SGM(blackbox_players, SCIP.Optimizer, max_iter=10)

    @test all([all(σ_bilateral .≈ σ_blackbox) for (σ_bilateral, σ_blackbox) in zip(Σ_bilateral, Σ_blackbox)])
    @test all([all(p_bilateral .≈ p_blackbox) for (p_bilateral, p_blackbox) in zip(poff_imp_bilateral, poff_imp_blackbox)])

    # not sure if necessary, but let's guarantee reproducibility
    IPG.initialize_strategies = IPG.initialize_strategies_feasibility
end

@testitem "Bilateral Payoff" begin
    quad_payoff = (xp, x_others) -> -xp[1]*xp[1] + xp[1] * x_others[1][1]
    Π_blackbox = BlackBoxPayoff(quad_payoff)
    Π_bilateral = QuadraticPayoff(0, [2, 1], 1)  # equivalent for the first player

    x = [[10.0], [10.0]]
    @test payoff(Π_blackbox, x[1], others(x, 1)) == payoff(Π_bilateral, x[1], others(x, 1))  # == 0.0
    x = [[0.0], [0.0]]
    @test payoff(Π_blackbox, x[1], others(x, 1)) == payoff(Π_bilateral, x[1], others(x, 1))
    x = [[10.0], [5.0]]
    @test payoff(Π_blackbox, x[1], others(x, 1)) == payoff(Π_bilateral, x[1], others(x, 1))
    σ = DiscreteMixedStrategy.(x)
    @test payoff(Π_blackbox, σ[1], others(σ, 1)) == payoff(Π_bilateral, σ[1], others(σ, 1))
    σ = [
        DiscreteMixedStrategy([0.5, 0.5], [[1], [0]]),
        DiscreteMixedStrategy([0.25, 0.75], [[5], [10]]),
    ]
    @test payoff(Π_blackbox, σ[1], others(σ, 1)) == payoff(Π_bilateral, σ[1], others(σ, 1))
end

@testitem "Player serialization" begin
    X1 = Model()
    @variable(X1, x1, start=10.0)
    @constraint(X1, x1 >= 0)

    player = Player(X1, QuadraticPayoff(0, [2, 1], 1), 1)

    filename = "test_player.json"
    IPG.save(player, filename)
    loaded_player = IPG.load(filename)

    @test loaded_player.p == player.p
    @test loaded_player.Π.cp == player.Π.cp
    @test loaded_player.Π.Qp == player.Π.Qp

    # test strategy space
    X1 = player.X
    loaded_X1 = loaded_player.X

    set_optimizer(X1, SCIP.Optimizer)
    set_optimizer(loaded_X1, SCIP.Optimizer)

    set_silent(X1)
    optimize!(X1)

    set_silent(loaded_X1)
    optimize!(loaded_X1)

    @test value.(all_variables(X1)) == value.(all_variables(loaded_X1))
    @test objective_value(X1) == objective_value(loaded_X1)

    # cleanup
    rm(filename)
end

@testitem "Example 5.3" begin
    using SCIP

    # guarantee reproducibility (always start with player 1)
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    P1 = Player(QuadraticPayoff(0, [2, 1], 1), 1)
    @variable(P1.X, x1, start=10.0)
    @constraint(P1.X, x1 >= 0)

    P2 = Player(QuadraticPayoff(0, [1, 2], 2), 2)
    @variable(P2.X, x2, start=10.0)
    @constraint(P2.X, x2 >= 0)

    Σ, payoff_improvements = IPG.SGM([P1, P2], SCIP.Optimizer, max_iter=5);

    @test [σ[1].supp for σ in Σ] ≈ [
        [[10.0]],
        [[10.0]],
        [[2.5]],
        [[2.5]],
        [[0.625]]
    ]
    @test [σ[2].supp for σ in Σ] ≈ [
        [[10.0]],
        [[5.0]],
        [[5.0]],
        [[1.25]],
        [[1.25]]
    ]
    @test all(Iterators.flatten(payoff_improvements) .≈ Iterators.flatten([
        (2, 25.0),
        (1, 56.25),
        (2, 14.0625),
        (1, 3.515625),
        (2, 0.87890625)
    ]))
end

@testitem "README Example Test" begin
    player_1 = Player(QuadraticPayoff(0, [2, 1], 1), 1)

    @variable(player_1.X, x1, start=10)
    @constraint(player_1.X, x1 >= 0)

    player_2 = Player(QuadraticPayoff(0, [1, 2], 2), 2)

    @variable(player_2.X, x2, start=10)
    @constraint(player_2.X, x2 >= 0)

    Σ, payoff_improvements = IPG.SGM([player_1, player_2], SCIP.Optimizer, max_iter=5)

    # Verify the final strategies match the expected values
    @test Σ[end][1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
    @test Σ[end][2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
end

@testitem "README Example Two-player" begin
    player_payoff(xp, x_others) = -(xp[1] * xp[1]) + xp[1] * prod(x_others[:][1])

    players = [
        Player(BlackBoxPayoff(player_payoff), 1),
        Player(BlackBoxPayoff(player_payoff), 2)
    ];

    @variable(players[1].X, x1, start=10); @constraint(players[1].X, x1 >= 0);

    @variable(players[2].X, x2, start=10); @constraint(players[2].X, x2 >= 0);

    Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5);

    # Verify the final strategies match the expected values
    @test Σ[end][1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
    @test Σ[end][2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
end
