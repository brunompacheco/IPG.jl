"""
Multiplayer quadratic game (qIPG).

Based on Dragotto and Scatamacchia (2023) description. For further details, see Section 5.4
of:
    [*G. Dragotto, R. Scatamacchia, "The Zero Regrets Algorithm: Optimizing over Pure Nash Equilibria via Integer Programming". 2023. INFORMS Journal on Computing*](https://doi.org/10.1287/ijoc.2022.0282)

# Notes
- SCIP can be replaced with other solvers, such as Gurobi or CPLEX.
"""

using IPG, IPG.JuMP, Random, LinearAlgebra, SCIP

"""
Generate a random qIPG game.

Based on the code repository accompanying the original paper. The original can be found in
https://github.com/gdragotto/ZeroRegretsAlgorithm
"""
function generate_random_instance(n::Int, m::Int, lower_bnd::Int, upper_bnd::Int; i_type="H")
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

    # build players and strategy spaces
    players = [Player() for _ in 1:n]
    for p in 1:n
        @variable(players[p].X, [1:m], Int, base_name="x_$p", lower_bound=lower_bnd, upper_bound=upper_bnd)
    end
    x = [[variable_by_name(players[p].X, "x_$p[$i]") for i in 1:m] for p in 1:n]

    # add payoffs
    for p in 1:n
        cp = rand(-RQ:RQ, m)
        
        # build payoff matrix
        Qp = Vector{Matrix{Float64}}()
        for k in 1:n
            push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
        end

        # build payoff expression
        Πp = cp'*x[p] + 0.5 * x[p]' * Qp[p] * x[p] + sum(x[k]' * Qp[k] * x[p] for k in 1:n if k != p)

        set_payoff!(players[p], Πp)
    end

    return players
end

players = generate_random_instance(3, 3, -500, 500)

IPG.initialize_strategies = IPG.initialize_strategies_feasibility

Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=25, verbose=true)
