using IPG
using Test
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
        Πp = QuadraticPayoff(cp, Qp)

        # build strategy space
        Xp = Model()
        @variable(Xp, lower_bound <= x[1:m] <= upper_bound, Int)

        push!(players, Player(Xp, Πp, p))
    end

    return players
end

@testset "IPG.jl" begin
    @testset "Two-player game" begin
        IPG.initialize_strategies = IPG.initialize_strategies_player_alone

        bilateral_players = generate_random_instance(2, 2, -5, 5)

        # "black-box" function for generic payoff players
        function generic_payoff(x, p)
            return sum([payoff(player.Πp, x, p) for player in bilateral_players])
        end

        generic_players = [
            Player(copy(player.Xp), GenericPayoff(generic_payoff), player.p)
            for player in bilateral_players
        ]

        Σ_bilateral, poff_imp_bilateral = IPG.SGM(bilateral_players, SCIP.Optimizer, max_iter=10)
        Σ_generic, poff_imp_generic = IPG.SGM(generic_players, SCIP.Optimizer, max_iter=10)

        @test all([all(σ_bilateral .≈ σ_generic) for (σ_bilateral, σ_generic) in zip(Σ_bilateral, Σ_generic)])
        @test all([all(p_bilateral .≈ p_generic) for (p_bilateral, p_generic) in zip(poff_imp_bilateral, poff_imp_generic)])

        # not sure if necessary, but let's guarantee reproducibility
        IPG.initialize_strategies = IPG.initialize_strategies_feasibility
    end

    @testset "Bilateral Payoff" begin
        quad_payoff = (x, p) -> -(x[p][1]*x[p][1]) + prod([x[i][1] for i in eachindex(x)])
        Π_generic = GenericPayoff(quad_payoff)
        Π_bilateral = QuadraticPayoff(0, [2, 1])  # equivalent for the first player

        x = [[10.0], [10.0]]
        @test payoff(Π_generic, x, 1) == payoff(Π_bilateral, x, 1)  # == 0.0
        x = [[0.0], [0.0]]
        @test payoff(Π_generic, x, 1) == payoff(Π_bilateral, x, 1)
        x = [[10.0], [5.0]]
        @test payoff(Π_generic, x, 1) == payoff(Π_bilateral, x, 1)
        σ = DiscreteMixedStrategy.(x)
        @test payoff(Π_generic, σ, 1) == payoff(Π_bilateral, σ, 1)
        σ = [
            DiscreteMixedStrategy([0.5, 0.5], [[1], [0]]),
            DiscreteMixedStrategy([0.25, 0.75], [[5], [10]]),
        ]
        @test payoff(Π_generic, σ, 1) == payoff(Π_bilateral, σ, 1)
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
