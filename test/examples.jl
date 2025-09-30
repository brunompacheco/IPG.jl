include("utils.jl")


@testitem "README Example Test" begin
    using IPG, SCIP

    # this is necessary for reproducibility, but doesn't affect the user experience
    IPG.get_player_order = IPG.get_player_order_fixed_descending

    P1 = Player()
    P2 = Player()

    @variable(P1.X, x1, start=10)

    @constraint(P1.X, x1 >= 0)

    @variable(P2.X, x2, start=10)

    @constraint(P2.X, x2 >= 0)

    set_payoff!(P1, -x1*x1 + x1*x2)
    @test string(P1.Π) == string(-x1*x1 + x1*x2)

    set_payoff!(P2, -x2*x2 + x1*x2)
    @test string(P2.Π) == string(-x2*x2 + x1*x2)

    Σ, payoff_improvements = SGM([P1, P2], SCIP.Optimizer, max_iter=5)

    # Verify the final strategies match the expected values
    @test Σ[end][P1] ≈ DiscreteMixedStrategy([1.0], [[0.625]])
    @test Σ[end][P2] ≈ DiscreteMixedStrategy([1.0], [[1.25]])
end

# The following tests on the examples/ should mostly guarantee that they run without errors.

@testitem "Example 5.3" begin
    include("../examples/example_5_3.jl")

    # TODO: this is currently our only (easy) way to check that an equilibrium was found.
    # And I'm not even sure that this is 100% reliable.
    @test length(payoff_improvements) == length(Σ) - 1
end

@testitem "Example CFLD" begin
    include("../examples/cfld.jl")

    @test length(payoff_improvements) <= length(Σ)
end

@testitem "Example qIPG" begin
    include("../examples/quad_game.jl")

    @test length(payoff_improvements) <= length(Σ)
end

@testitem "Example Selfish EBMC" begin
    using Downloads

    url = "https://raw.githubusercontent.com/HyunwooLee0429/Best-response-dynamics-IPG/main/BZR_EBMC/EBMC_generated/single_dataset/2_50_0.3.csv"
    target_dir = joinpath(@__DIR__, "..", "examples", "EBMC_generated", "single_dataset")
    mkpath(target_dir)
    target_file = joinpath(target_dir, "2_50_0.3.csv")

    if !isfile(target_file)
        try
            Downloads.download(url, target_file)
        catch e
            @warn "Could not download EBMC CSV" exception = e url = url
        end
    end

    include("../examples/ebmc.jl")

    # POS has to be >= 1
    @test objective_value(model_sw) / sw_ne >= 1.0
end