using CSV
using DataFrames
using PyCall

py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

load_pickle = py"load_pickle"

# these parameters must match those in the csv file name, and the folder from which they are downloaded
type_dataset = "single"
county_size = 2
num_lakes_per_county = 50
budget_ratio = 0.3

dirname = "EBMC_generated/$(type_dataset)_dataset/"
fname = "$(county_size)_$(num_lakes_per_county)_$(budget_ratio).csv"
df_edge = DataFrame(CSV.File(dirname * fname))

info_data = load_pickle(dirname * "info_data.pickle")

# === Unpack Experiment Settings === #
# extract the value list for the (county_size, num_lakes_per_county, budget_ratio) key
vals = info_data[(county_size, num_lakes_per_county, budget_ratio)]

# values come from Python (0-based originally) so use 1-based Julia indices
counties = vals[1]                 # likely a sequence of county ids
num_lakes_per_county = Int(vals[2])         # ensure it's an Int
infestation_status = vals[3]             # likely a Python dict
county_budget = vals[4]

# determine infested lakes (any value > 0 in the nested dict)
infested_lakes = String[]
for (key, infestation_vals) in infestation_status
    if any(v -> v > 0, values(infestation_vals))
        push!(infested_lakes, string(key))
    end
end

# === lakes and lake->county mapping === #
lakes = unique(vcat(df_edge[:, :dow_origin], df_edge[:, :dow_destination]))
lake_county = Dict(lake => lake[1:2] for lake in lakes)   # first two chars like Python's [:2]

# === Compute Lake Weights === #
w = Dict{String, Float64}(lake => 0.0 for lake in lakes)
for row in eachrow(df_edge)
    if row[:bij] != 0
        ori = string(row[:dow_origin])
        dst = string(row[:dow_destination])
        w[ori] += row[:bij] * row[:weight]
        w[dst] += row[:bij] * row[:weight]
    end
end

# === Set Model Parameters === #
I = lakes

I_c = Dict(county => [i for i in I if i[1:2] == county] for county in counties)
I_c_complement = Dict(county => [i for i in I if !(i in I_c[county])] for county in counties)

# build arc dictionaries n (weight) and t (bij)
n = Dict{Tuple{Any,Any}, Float64}()
t = Dict{Tuple{Any,Any}, Float64}()
for row in eachrow(df_edge)
    arc = (row[:dow_origin], row[:dow_destination])
    n[arc] = row[:weight]
    t[arc] = row[:bij]
end

arcs = collect(keys(n))

# arcs within, incoming to, and outgoing from each county
arcs_c = Dict{Any, Vector{Tuple{Any,Any}}}()
arcs_plus_c = Dict{Any, Vector{Tuple{Any,Any}}}()
arcs_minus_c = Dict{Any, Vector{Tuple{Any,Any}}}()

for county in counties
    arcs_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] == county)]
    arcs_plus_c[county] = [arc for arc in arcs if (arc[2][1:2] == county) && (arc[1][1:2] != county)]
    arcs_minus_c[county] = [arc for arc in arcs if (arc[1][1:2] == county) && (arc[2][1:2] != county)]
end


# === Solve the Social Welfare Model === #
using IPG.JuMP, SCIP

model_sw = Model(SCIP.Optimizer)
set_silent(model_sw)
@variable(model_sw, x_sw[I], Bin)
@variable(model_sw, y_sw[arcs], Bin)  # auxiliary variable for convenience
@constraint(model_sw, [arc in arcs], y_sw[arc] <= x_sw[arc[1]] + x_sw[arc[2]])
@constraint(model_sw, [county in counties], sum(x_sw[i] for i in I_c[county]) <= county_budget[county])

@objective(model_sw, Max, sum(t[arc] * n[arc] * y_sw[arc] for arc in arcs))

optimize!(model_sw)

println("Optimal Social Welfare: ", objective_value(model_sw))
osw_val = value.(x_sw)


# === Define and Solve SELFISH Game using IPG.jl === #
using IPG

# define players
players = [Player(name=county) for county in counties]

# add variables
x_c = Dict(p => @variable(p.X, [I_c[p.name]], Bin, base_name="x_$(p.name)_") for p in players)

# concatenate x variables
x = Containers.DenseAxisArray(vcat([x_c[p].data for p in players]...), vcat([x_c[p].axes[1] for p in players]...))

# warm start from social welfare solution
# for i in lakes
#     set_start_value(x[i], value(x_sw[i]))
# end

# y_ij = x_i ∨ xj
y = Dict(arc => x[arc[1]] + x[arc[2]] - x[arc[1]] * x[arc[2]] for arc in arcs)  # auxiliary variable for convenience

for p in players
    ### add constraints
    @constraint(p.X, sum(x_c[p]) <= county_budget[p.name])

    ### set payoff
    set_payoff!(p, sum(t[arc] * n[arc] * y[arc] for arc in arcs_minus_c[p.name]))
end

Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=10, verbose=true)
σ_ne = Σ[end]

# compute social welfare
function social_welfare(x_c_val)
    x_val = Dict(I_c[p.name][i] => x_c_val[p][i] for p in players for i in eachindex(I_c[p.name]))

    y_val = Dict(arc => x_val[arc[1]] + x_val[arc[2]] - x_val[arc[1]] * x_val[arc[2]] for arc in arcs)

    sw = 0
    for arc in arcs
        sw += t[arc] * n[arc] * y_val[arc]
    end

    return sw
end

if all(length(σ_ne[p].probs) == 1 for p in players)
    println("Pure NE found.")

    x_ne = first(IPG.support(σ_ne))

    sw_ne = social_welfare(x_ne)
    println("PNE social welfare: ", sw_ne)
    println("POS: ", objective_value(model_sw) / sw_ne)
else
    expected_social_welfare = expected_value(social_welfare, σ_ne)

    println("Expected social welfare from MNE: ", expected_social_welfare)
    println("POS: ", objective_value(model_sw) / expected_social_welfare)

    error("Mixed strategy NE found!")
end
