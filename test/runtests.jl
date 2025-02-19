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
