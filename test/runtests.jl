using IPG
using Test
using LinearAlgebra, JuMP, SCIP

""" The random instance generation procedure is based on
> Dragotto, Gabriele, and Rosario Scatamacchia. “The Zero Regrets Algorithm: Optimizing over Pure Nash Equilibria via Integer Programming.” INFORMS Journal on Computing 35, no. 5 (September 2023): 1143–60. https://doi.org/10.1287/ijoc.2022.0282.

The original can be seen in their repository: https://github.com/gdragotto/ZeroRegretsAlgorithm
"""
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

    # build players' parameters
    players = []
    for p in 1:n
        # build payoff
        Qp = Vector{Matrix{Float64}}()
        for k in 1:n
            push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
        end

        cp = rand(-RQ:RQ, m)
        push!(players, [cp, Qp, p, m, lower_bound, upper_bound])
    end

    return players
end

@testset "IPG.jl" begin
    @testset "Two-player game" begin
        IPG.initialize_strategies = IPG.initialize_strategies_player_alone

        params = generate_random_instance(2, 2, -5, 5)

        # build bilateral players
        bilateral_players = Vector{Player{QuadraticPayoff}}()
        for (cp, Qp, p, m, lower_bound, upper_bound) in params
            Πp = QuadraticPayoff(cp, Qp, p)

            # build strategy space
            Xp = Model()
            @variable(Xp, lower_bound <= xp[1:m] <= upper_bound, Int)
            set_start_value.(xp, 1.0)

            push!(bilateral_players, Player(Xp, xp, Πp, p))
        end

        # "black-box" function for blackbox payoff players
        function get_blackbox_function(p)
            return (xp, x_others) -> payoff(bilateral_players[p].Π, xp, x_others)
        end

        # build blackbox players
        blackbox_players = Vector{Player{BlackBoxPayoff}}()
        for (cp, Qp, p, m, lower_bound, upper_bound) in params
            Πp = BlackBoxPayoff((xp, x_others) -> payoff(QuadraticPayoff(cp, Qp, p), xp, x_others))

            # build strategy space
            Xp = Model()
            @variable(Xp, lower_bound <= xp[1:m] <= upper_bound, Int)
            set_start_value.(xp, 1.0)

            push!(blackbox_players, Player(Xp, xp, Πp, p))
        end

        Σ_bilateral, poff_imp_bilateral = IPG.SGM(bilateral_players, SCIP.Optimizer, max_iter=10)
        Σ_blackbox, poff_imp_blackbox = IPG.SGM(blackbox_players, SCIP.Optimizer, max_iter=10)

        @test all([all(σ_bilateral .≈ σ_blackbox) for (σ_bilateral, σ_blackbox) in zip(Σ_bilateral, Σ_blackbox)])
        @test all([all(p_bilateral .≈ p_blackbox) for (p_bilateral, p_blackbox) in zip(poff_imp_bilateral, poff_imp_blackbox)])

        # not sure if necessary, but let's guarantee reproducibility
        IPG.initialize_strategies = IPG.initialize_strategies_feasibility
    end

    @testset "Bilateral Payoff" begin
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

    @testset "DiscreteMixedStrategy" begin
        σp = DiscreteMixedStrategy([0.5, 0.5], [[1, 0], [0, 1]])
        @test length(σp.probs) == 2
        @test size(σp.supp) == (2,)
        @test size(σp.supp[1]) == (2,)
        @test size(σp.supp[2]) == (2,)
        @test IPG.is_pure(σp) == false
        @test expected_value(identity, σp) == [0.5, 0.5]
        @test expected_value(sum, σp) == 1.0

        xp = [1, 2, 3]
        σp = DiscreteMixedStrategy(xp)
        @test expected_value(identity, σp) == xp

        σa = DiscreteMixedStrategy([0.5, 0.5], [[1, 0], [0, 1]])
        σb = DiscreteMixedStrategy([0.5, 0.5], [[1, 0], [0, 1]])
        @test σa == σb

        σc = DiscreteMixedStrategy([0.5 + eps(), 0.5 - eps()], [[1, 0], [0, 1]])
        @test σa != σc
        @test σa ≈ σc
    end

    @testset "Player serialization" begin
        # WARNING: support for serialization has been removed
        # X1 = Model()
        # @variable(X1, x1, start=10.0)
        # @constraint(X1, x1 >= 0)

        # player = Player(X1, [x1], QuadraticPayoff(0, [2, 1], 1), 1)

        # filename = "test_player.json"
        # IPG.save(player, filename)
        # loaded_player = IPG.load(filename)

        # @test loaded_player.p == player.p
        # @test loaded_player.Π.cp == player.Π.cp
        # @test loaded_player.Π.Qp == player.Π.Qp

        # # test strategy space
        # X1 = player.X
        # loaded_X1 = loaded_player.X

        # set_optimizer(X1, SCIP.Optimizer)
        # set_optimizer(loaded_X1, SCIP.Optimizer)

        # set_silent(X1)
        # optimize!(X1)

        # set_silent(loaded_X1)
        # optimize!(loaded_X1)

        # @test value.(player.vars) == value.(loaded_player.vars)
        # @test objective_value(X1) == objective_value(loaded_X1)

        # # cleanup
        # rm(filename)
    end

    @testset "Example 5.3" begin
        using SCIP

        # guarantee reproducibility (always start with player 1)
        IPG.get_player_order = IPG.get_player_order_fixed_descending

        X1 = Model()
        @variable(X1, x1, start=10.0)
        @constraint(X1, x1 >= 0)

        P1 = Player(X1, [x1], QuadraticPayoff(0, [2, 1], 1), 1)

        X2 = Model()
        @variable(X2, x2, start=10.0)
        @constraint(X2, x2 >= 0)
        P2 = Player(X2, [x2], QuadraticPayoff(0, [1, 2], 2), 2)

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

    @testset "README Example Test" begin
        X1 = Model()
        @variable(X1, x1, start=10)
        @constraint(X1, x1 >= 0)

        Π1 = QuadraticPayoff(0, [2, 1], 1)

        player_1 = Player(X1, [x1], Π1, 1)

        X2 = Model()
        @variable(X2, x2, start=10)
        @constraint(X2, x2 >= 0)

        Π2 = QuadraticPayoff(0, [1, 2], 2)

        player_2 = Player(X2, [x2], Π2, 2)

        Σ, payoff_improvements = IPG.SGM([player_1, player_2], SCIP.Optimizer, max_iter=5)

        # Verify the final strategies match the expected values
        @test Σ[end][1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
        @test Σ[end][2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
    end

    @testset "README Example Two-player" begin
        player_payoff(xp, x_others) = -(xp[1] * xp[1]) + xp[1] * prod(x_others[:][1])

        X1 = Model(); X2 = Model();

        @variable(X1, x1, start=10); @constraint(X1, x1 >= 0);
        @variable(X2, x2, start=10); @constraint(X2, x2 >= 0);

        players = [
            Player(X1, [x1], BlackBoxPayoff(player_payoff), 1),
            Player(X2, [x2], BlackBoxPayoff(player_payoff), 2)
        ];

        Σ, payoff_improvements = IPG.SGM(players, SCIP.Optimizer, max_iter=5);

        # Verify the final strategies match the expected values
        @test Σ[end][1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
        @test Σ[end][2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
    end
end
