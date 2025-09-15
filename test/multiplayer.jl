include("utils.jl")

@testitem "Construction of normal game" setup=[Utilities] begin
    players = get_example_two_player_game()

    S_X = Sample(players[1] => [[10.0],[5.0]], players[2]=> [[1.0],[5.0]])
    sampled_game = IPG.UtilitiesSampledGame(S_X)

    σ = IPG.solve_multigame(sampled_game, nothing)

    @test expected_value(identity, σ[players[1]]) == [5.0]
    @test expected_value(identity, σ[players[2]]) == [1.0]
end
