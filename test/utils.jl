using IPG
using TestItems


@testsnippet Utilities begin
    using LinearAlgebra, JuMP, SCIP

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

        # build strategy spaces
        X = [Model() for _ in 1:n]
        for p in 1:n
            @variable(X[p], [1:m], Int, base_name="x_$p", lower_bound=lower_bnd, upper_bound=upper_bnd)
        end
        x = [[variable_by_name(X[p], "x_$p[$i]") for i in 1:m] for p in 1:n]

        # build players
        players = Vector{Player}()
        for p in 1:n
            # build payoff
            Qp = Vector{Matrix{Float64}}()
            for k in 1:n
                push!(Qp, M[((p-1) * m + 1):(p * m), ((k-1) * m + 1):(k * m)])
            end

            cp = rand(-RQ:RQ, m)
            
            Πp = cp'*x[p]
            Πp += 0.5 * x[p]' * Qp[p] * x[p]
            Πp += sum(x[k]' * Qp[k] * x[p] for k in 1:n if k != p)

            # build strategy space
            push!(players, Player(X[p], Πp))
        end

        return players
    end

    function get_example_two_player_game()
        # Example 5.3 from the IPG paper
        X1 = Model()
        @variable(X1, x1, start=10.0)
        @constraint(X1, x1 >= 0)

        X2 = Model()
        @variable(X2, x2, start=10.0)
        @constraint(X2, x2 >= 0)

        function player_payoff(x_self, x_other)
            return -x_self * x_self + x_self * x_other
        end

        return [
            Player(X1, player_payoff(x1, x2)),
            Player(X2, player_payoff(x2, x1))
        ]
    end
end
