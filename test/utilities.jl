include("utils.jl")

@testitem "Computation of utilities tensor" setup=[Utilities] begin
    p1, p2 = get_example_two_player_game()

    # give some options so that we can test the utility tensor
    S_X = IPG.Sample{PureStrategy}(p1 => [[10.0],[5.0]], p2 => [[10.0],[5.0]])

    utilities = IPG.get_utilities([p1,p2], S_X)

    println(size(utilities))
    expected_utilities = zeros(Float64, 2, 2, 2)  # 2 strategies for p1, 2 strategies for p2, 2 players
    expected_utilities[1,1,:] .= expected_utilities[2,2,:] .= 0.0
    expected_utilities[1,2,1] = expected_utilities[2,1,2] = -50.0
    expected_utilities[2,1,1] = expected_utilities[1,2,2] = 25.0
    @test utilities == expected_utilities
end

@testitem "Update of utilities tensor" setup=[Utilities] begin
    p1, p2 = get_example_two_player_game()

    # give some options so that we can test the utility tensor
    S_X = IPG.Sample{PureStrategy}(p1 => [[10.0],[5.0]], p2 => [[10.0]])

    utilities = IPG.get_utilities([p1,p2], S_X)

    expected_utilities = zeros(Float64, 2, 2, 2)  # 2 strategies for p1, 2 strategies for p2, 2 players
    expected_utilities[1,1,:] .= expected_utilities[2,2,:] .= 0.0
    expected_utilities[1,2,1] = expected_utilities[2,1,2] = -50.0
    expected_utilities[2,1,1] = expected_utilities[1,2,2] = 25.0

    @test utilities == expected_utilities[:,1:1,:]

    push!(S_X[p2], [5.0])

    utilities = IPG.update_utilities(utilities, [p1, p2], S_X)

    @test utilities == expected_utilities
end