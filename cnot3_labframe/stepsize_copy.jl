#==========================================
# Copy of the data collection functions from the cnot3 stepsize test
==========================================#
using QuantumGateDesign, DelimitedFiles


function collect_data(prob::SchrodingerProb, controls::ControlsType,
        pcof::AbstractVector{<: Real}, order::Integer, max_walltime::Real,
        filename_base::AbstractString, N_timestep_saves::Integer, initial_nsteps=2
    )
    prob = copy(prob) # Copy problem, just to make sure there are no mutability issues.

    initial_time = time()
    max_walltime *= 60*60 # convert walltime from hours to seconds
    
    filename_csv = filename_base * ".csv"
    filename_final_states_csv = filename_base * "_finalStates.csv"

    header = hcat(
        "nsteps", "stepsize", "elapsed_time", "avg_N_gmres_iter",
        "N_converged_gmres", "avg_gmres_residual", "max_gmres_residual",
        "R_abs_err_L1", "R_abs_err_L2", "R_rel_err_L1", "R_rel_err_L2",
        "R_abs_err_Linf",
    )
    println(stdout, header)

    writedlm(filename_csv, rpad.(header, 24), ',')

    final_states_vec = Vector{ComplexF64}(undef, length(prob.u0))
    final_states_vec .= NaN
    gmres_tracker = GMRESTracker()

    # Run simulation
    prob.nsteps = initial_nsteps
    stepsize = prob.tf / prob.nsteps

    # Run simulation once just to get compilation out of the way
    dummy_history = eval_forward(prob, controls, pcof, order=order)

    is_first_step = true
    history_2h = nothing
    history_h = nothing
    elapsed_time = 0.0

    # Loop until time runs out (with estimator for when we will go overtime on next simulation)
    while (time()-initial_time) < (max_walltime - 2*elapsed_time)
        # Run simulation
        t1 = time()
        history_h = eval_forward(prob, controls, pcof, order=order,
                                 gmres_tracker=gmres_tracker, verbose=true)
        t2 = time()
        elapsed_time = t2 - t1

        # Collect data into a row
        if !is_first_step
            R = RichardsonExtrapolation(history_h[:,1:2:end,:], history_2h, order)
            csv_row = hcat(
                prob.nsteps, stepsize, elapsed_time,
                avg_N_iterations(gmres_tracker), gmres_tracker.N_converged,
                avg_residual(gmres_tracker), R.abs_err_L1, R.abs_err_L2,
                R.rel_err_L1, R.rel_err_L2, R.abs_err_Linf,
            )
        else
            csv_row = hcat(
                prob.nsteps, stepsize, elapsed_time,
                avg_N_iterations(gmres_tracker), gmres_tracker.N_converged,
                avg_residual(gmres_tracker), NaN, NaN, NaN, NaN, NaN, 
            )
            is_first_step = false
        end

        # Log data (CSV)
        open(filename_csv, "a+") do io
            writedlm(io, rpad.(csv_row, 24), ',')
        end
        final_states_vec .= reshape(history_h[:,end,:], :)
        open(filename_final_states_csv, "a+") do io
            writedlm(io, rpad.(transpose(final_states_vec), 53), ',')
        end
        println(stdout, csv_row) # Print row

        # Prepare for next iteration
        history_2h = history_h
        prob.nsteps *= 2
        stepsize = prob.tf / prob.nsteps
    end

    return readdlm(filename_csv, ',')
end


function collect_data_grad(prob::SchrodingerProb, controls::ControlsType,
        pcof::AbstractVector{<: Real}, target::AbstractMatrix{<: Number}, order::Integer, max_walltime::Real,
        filename_base::AbstractString, N_timestep_saves::Integer
    )
    prob = copy(prob) # Copy problem, just to make sure there are no mutability issues.

    initial_time = time()
    max_walltime *= 60*60 # convert walltime from hours to seconds
    
    filename_csv = filename_base * ".csv"
    filename_final_states_csv = filename_base * "_finalStates.csv"

    header = hcat(
        "nsteps", "stepsize", "elapsed_time", "forward_time", "adjoint_time",
        "grad_accum_time", "avg_N_gmres_iter_fwd", "N_converged_gmres_fwd",
        "avg_gmres_residual_fwd", "avg_N_gmres_iter_adj", "N_converged_gmres_adj",
        "avg_gmres_residual_adj", "R_abs_err_L1", "R_abs_err_L2", "R_rel_err_L1",
        "R_rel_err_L2", "R_abs_err_Linf",
    )
    println(stdout, header)

    writedlm(filename_csv, rpad.(header, 24), ',')

    final_states_vec = Vector{ComplexF64}(undef, length(prob.u0))
    final_states_vec .= NaN

    forward_gmres_tracker = GMRESTracker()
    adjoint_gmres_tracker = GMRESTracker()

    # Run simulation
    prob.nsteps = 2
    stepsize = prob.tf / prob.nsteps

    # Run simulation once just to get compilation out of the way
    dummy_history = eval_forward(prob, controls, pcof, order=order)

    timer = QuantumGateDesign.DiscreteAdjointTimes()
    is_first_step = true
    history_2h = nothing
    history_h = nothing
    elapsed_time = 0.0
    N_derivatives = div(order,2)
    grad = zeros(get_number_of_control_parameters(controls))

    # Loop until time runs out (with estimator for when we will go overtime on next simulation)
    while (time()-initial_time) < (max_walltime - 2*elapsed_time)
        # Run simulation
        history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)
        lambda_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)
        adjoint_forcing = zeros(prob.real_system_size, 1+prob.nsteps, prob.N_initial_conditions)

        QuantumGateDesign.discrete_adjoint!(
            grad, history, lambda_history, adjoint_forcing, prob, controls,
            pcof, target, order=order, timer=timer,
            forward_gmres_tracker=forward_gmres_tracker,
            adjoint_gmres_tracker=adjoint_gmres_tracker,
        )
        history_h = QuantumGateDesign.real_to_complex(history[:,1,:,:])

        # Collect data into a row
        if !is_first_step
            R = RichardsonExtrapolation(history_h[:,1:2:end,:], history_2h, order)
            csv_row = hcat(
                prob.nsteps, stepsize, QuantumGateDesign.total_time(timer),
                timer.forward, timer.adjoint, timer.grad_accum,
                avg_N_iterations(forward_gmres_tracker),
                forward_gmres_tracker.N_converged,
                avg_residual(forward_gmres_tracker),
                avg_N_iterations(adjoint_gmres_tracker),
                adjoint_gmres_tracker.N_converged,
                avg_residual(adjoint_gmres_tracker), R.abs_err_L1,
                R.abs_err_L2, R.rel_err_L1, R.rel_err_L2, R.abs_err_Linf,
            )
        else
            # Rerun to update timing (now that precompilation is out of the way)
            QuantumGateDesign.discrete_adjoint!(
                grad, history, lambda_history, adjoint_forcing, prob, controls,
                pcof, target, order=order, timer=timer,
                forward_gmres_tracker=forward_gmres_tracker,
                adjoint_gmres_tracker=adjoint_gmres_tracker,
            )

            csv_row = hcat(
                prob.nsteps, stepsize, QuantumGateDesign.total_time(timer),
                timer.forward, timer.adjoint, timer.grad_accum,
                avg_N_iterations(forward_gmres_tracker),
                forward_gmres_tracker.N_converged,
                avg_residual(forward_gmres_tracker),
                avg_N_iterations(adjoint_gmres_tracker),
                adjoint_gmres_tracker.N_converged,
                avg_residual(adjoint_gmres_tracker), NaN,
                NaN, NaN, NaN, NaN,
            )
            is_first_step = false
        end

        # Log data (CSV)
        open(filename_csv, "a+") do io
            writedlm(io, rpad.(csv_row, 24), ',')
        end
        final_states_vec .= reshape(history_h[:,end,:], :)
        open(filename_final_states_csv, "a+") do io
            writedlm(io, rpad.(transpose(final_states_vec), 53), ',')
        end
        println(stdout, csv_row) # Print row

        # Prepare for next iteration
        history_2h = history_h
        prob.nsteps *= 2
        stepsize = prob.tf / prob.nsteps
    end

    return readdlm(filename_csv, ',')
end



"""
Turn state vector history into a row matrix with the correct number of timesteps
saved, for putting into a csv file. 

If the number of timesteps to be saved is greater than the number of timesteps
in the provided history, use NaN in place of the 'missing' timestep saves
"""
function parse_history_for_csv(history::AbstractArray{ComplexF64, 3}, N_timestep_saves::Union{Missing, Integer})
    if ismissing(N_timestep_saves)
        return copy(reshape(history, 1, :))
    end

    parsed_history = Array{ComplexF64, 3}(undef, size(history, 1), 1+N_timestep_saves, size(history, 3))
    parsed_history .= NaN

    nsteps = size(history, 2) - 1

    if  nsteps >= N_timestep_saves
        if (nsteps % N_timestep_saves != 0)
            throw(ArgumentError("Number of timesteps to save ($N_timestep_saves) does not divide the number of timesteps ($nsteps)"))
        end
        stride = div(nsteps, N_timestep_saves)
        parsed_history .= history[:,1:stride:end,:]
    else
        if (N_timestep_saves % nsteps != 0)
            throw(ArgumentError("Number of timesteps ($nsteps) does not divide the number of timesteps to save ($N_timestep_saves)"))
        end
        stride = div(N_timestep_saves, nsteps)
        parsed_history[:,1:stride:end,:] .= history
    end

    return reshape(parsed_history, 1, :)
end
