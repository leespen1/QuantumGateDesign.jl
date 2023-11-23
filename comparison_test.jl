using Revise
using HermiteOptimalControl
using JLD2
include("examples/SWAP_example.jl")

prob, control, pcof, target, pcof_u, pcof_l= main(d=1, N_guard=1, tf=50.0, D1=5)
f = jldopen("history_comparison.jld2", "r")
pcof = copy(f["pcof"])
grad, history, lambda_history = discrete_adjoint(prob, control, pcof, target, order=4, return_lambda_history=true)