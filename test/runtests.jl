using IPG
using Test

@testset "IPG.jl" begin
    @testset "DiscreteMixedStrategy" begin
        σp = DiscreteMixedStrategy([0.5, 0.5], [[1, 0], [0, 1]])
        @test length(σp.probs) == 2
        @test size(σp.supp) == (2,)
        @test size(σp.supp[1]) == (2,)
        @test size(σp.supp[2]) == (2,)
        @test IPG.is_pure(σp) == false
        @test expected_value(identity, σp) == [0.5, 0.5]
        @test expected_value(sum, σp) == 1.0
    end

    @testset "Player serialization" begin
        using JuMP, SCIP

        X1 = Model()
        @variable(X1, x1, start=10.0)
        @constraint(X1, x1 >= 0)

        player = Player(X1, QuadraticPayoff(0, [2, 1]), 1)

        filename = "test_player.json"
        IPG.save(player, filename)
        loaded_player = IPG.load(filename)

        @test loaded_player.p == player.p
        @test loaded_player.Πp.cp == player.Πp.cp
        @test loaded_player.Πp.Qp == player.Πp.Qp

        # test strategy space
        X1 = player.Xp
        loaded_X1 = loaded_player.Xp

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

    @testset "Example 5.3" begin
        using SCIP

        # guarantee reproducibility (always start with player 1)
        IPG.get_player_order = IPG.get_player_order_fixed_descending

        P1 = Player(QuadraticPayoff(0, [2, 1]), 1)
        @variable(P1.Xp, x1, start=10.0)
        @constraint(P1.Xp, x1 >= 0)

        P2 = Player(QuadraticPayoff(0, [1, 2]), 2)
        @variable(P2.Xp, x2, start=10.0)
        @constraint(P2.Xp, x2 >= 0)

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
end
