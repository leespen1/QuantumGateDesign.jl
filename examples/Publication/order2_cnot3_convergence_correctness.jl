include(joinpath(@__DIR__, "setup_prob_cnot3_convergence_correctness.jl"))

# Get test start time
current_date_time = Dates.now()
formatted_date_time = Dates.format(current_date_time, "yyyy-mm-dd_HH:MM:SS")
jld2_filename = "order2_cnot3_convergence_correctness_$formatted_date_time.jld2"
N_iterations = 19 #32
ret_qgd_order2 = QuantumGateDesign.get_histories(
    prob, controls_autodiff, pcof, N_iterations, 
    abstol=gmres_abstol, reltol=gmres_abstol, orders=[2],
    jld2_filename=jld2_filename
)

println("Finished Juqbox")
