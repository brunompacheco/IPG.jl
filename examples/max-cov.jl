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
type_dataset = "multi"
county_size = 5
num_lakes_per_county = 50
budget_ratio = 0.5

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

# === Define and Solve SELFISH Game using IPG.jl === #

using IPG, SCIP
using IPG.JuMP: Containers

# define players
players = [Player(name=county) for county in counties]

# add variables
x_c = Dict(p => @variable(p.X, [I_c[p.name]], Bin, base_name="x_$(p.name)_") for p in players)

# concatenate x variables
x = Containers.DenseAxisArray(vcat([x_c[p].data for p in players]...), vcat([x_c[p].axes[1] for p in players]...))

y = Dict(arc => x[arc[1]] + x[arc[2]] for arc in arcs)  # auxiliary variable for convenience

for p in players
    ### add constraints
    @constraint(p.X, sum(x_c[p]) <= county_budget[p.name])

    ### set payoff
    set_payoff!(p, sum(t[arc] * n[arc] * y[arc] for arc in arcs_minus_c[p.name]))
end

Σ, payoff_improvements = SGM(players, SCIP.Optimizer, max_iter=10, verbose=true)
