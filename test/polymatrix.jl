include("utils.jl")

@testitem "Polymatrix computation" setup=[Utilities] begin
    players = get_example_two_player_game()
    for player in players
        IPG.set_optimizer(player, SCIP.Optimizer)
    end

    # give some options so that we can test the polymatrix
    S_X = Sample(players[1] => [[10.0],[5.0]], players[2]=> [[10.0],[5.0]])

    polymatrix = IPG.get_polymatrix_bilateral(S_X)

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

    two_player_polymatrix = IPG.get_polymatrix_twoplayers(S_X)

    @test two_player_polymatrix == polymatrix

    incremental_S_X = IPG.initialize_strategies(players)  # initialized from start values
    sampled_game = IPG.PolymatrixSampledGame(incremental_S_X)
    IPG.add_new_strategy!(sampled_game, players[1], [5.0])
    IPG.add_new_strategy!(sampled_game, players[2], [5.0])

    @test sampled_game.polymatrix == polymatrix
end

@testitem "Solving polymatrix game" setup=[Utilities] begin
    players = get_example_two_player_game()

    S_X = Sample(players[1] => [[10.0],[5.0]], players[2]=> [[1.0],[5.0]])

    sampled_game = IPG.PolymatrixSampledGame(S_X)

    σ_PNS = IPG.solve_PNS(sampled_game, SCIP.Optimizer)
    σ_Sandholm = IPG.solve_Sandholm1(sampled_game, SCIP.Optimizer)

    @test expected_value(identity, σ_PNS[players[1]]) == expected_value(identity, σ_Sandholm[players[1]]) == [5.0]
    @test expected_value(identity, σ_PNS[players[2]]) == expected_value(identity, σ_Sandholm[players[2]]) == [1.0]
end
