include(joinpath(@__DIR__, "setup_prob_cnot3_convergence_correctness.jl"))

# Get test start time
current_date_time = Dates.now()
formatted_date_time = Dates.format(current_date_time, "yyyy-mm-dd_HH:MM:SS")
jld2_filename = "juqbox_cnot3_convergence_correctness_$formatted_date_time.jld2"
N_iterations = 19 #32
ret_juq = QuantumGateDesign.get_histories(params, wa, pcof, N_iterations,
                                          jld2_filename=jld2_filename)

println("Finished Juqbox")
