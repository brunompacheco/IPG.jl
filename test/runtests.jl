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
end
