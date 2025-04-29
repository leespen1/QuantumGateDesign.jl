using QuantumGateDesign, DelimitedFiles, Dates, Random, LinearAlgebra, ArgParse, Juqbox

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--atol", "-a"
            help = "Absolute tolerance to use in the linear solves."
            arg_type = Float64
            default = 1e-10
        "--rtol", "-r"
            help = "Relative tolerance to use in the linear solves."
            arg_type = Float64
            default = 1e-12
        "--D1"
            help = "D1, control number of control parameters."
            arg_type = Int64
            default = 15
        "--time", "-t"
            help = "Amount of wall time (in hours) to spend on test."
            arg_type = Float64
            default = 1 # Default
        "--nsaves", "-s"
            help = "Number of timesteps to save in CSV files."
            arg_type = Int64
            default = 64 # Default
        "--output_directory", "-o"
            help = "Directory to store data."
            arg_type = String
            default = "Data"
        "--use_juqbox"
            help = "Use Juqbox to perform the timestepping."
            arg_type = Bool
            default = false
        "--gradient"
            help = "Calculate gradient in addition to forward evolution."
            arg_type = Bool
            default = true
        "--levels_cavity", "-l"
            help = "Number of energy levels to use for the cavity in the Hamiltonian model."
            arg_type = Int64
            default = 4
        "--gate_duration", "-d"
            help = "Duration of the gate, in nanoseconds."
            arg_type = Float64
            default = 550.0
        "--start_in_highest_state"
            help = "Instead of doing the typical gate design problem, start in the most excited state, which should have the fastest dynamics."
            arg_type = Bool
            default = false
        "order"
            help = "Method order to use"
            required = true
            arg_type = Int64
        "degree"
            help = "Degree of B-spline to use"
            required = true
            arg_type = Int64
        "seed"
            help = "Seed to use when generating control vector."
            required = true
            arg_type = Int64

    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    order = parsed_args["order"]
    degree = parsed_args["degree"]
    seed = parsed_args["seed"]
    atol = parsed_args["atol"]
    rtol = parsed_args["rtol"]
    D1 = parsed_args["D1"]
    time = parsed_args["time"]
    N_timestep_saves = parsed_args["nsaves"]
    output_directory = parsed_args["output_directory"]
    use_juqbox = parsed_args["use_juqbox"]
    compute_gradient = parsed_args["gradient"]
    N_osc_levels = parsed_args["levels_cavity"]
    Tmax = parsed_args["gate_duration"]
    start_in_highest_state = parsed_args["start_in_highest_state"]
    nthreads = Threads.nthreads()

    cnot3ret = QuantumGateDesign.setup_cnot3(
        seed=seed,
        atol=atol,
        rtol=rtol,
        D1=D1,
        N_osc_levels=N_osc_levels,
        Tmax=Tmax
    )


    if start_in_highest_state # Change initial conditions to start in highest state
        cnot3ret.qgd_prob.u0 = zeros(cnot3ret.qgd_prob.N_tot_levels, 1)
        cnot3ret.qgd_prob.v0 = zeros(cnot3ret.qgd_prob.N_tot_levels, 1)
        cnot3ret.qgd_prob.u0[end,1] = 1
        cnot3ret.qgd_prob.N_ess_levels = 1
        cnot3ret.qgd_prob.N_initial_conditions = 1

        cnot3ret.target = zeros(cnot3ret.qgd_prob.N_tot_levels, 1)
        cnot3ret.target[end,1] = 1
    end

    display(cnot3ret.qgd_prob)


    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between amax and -amax
    pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))

    mkpath(output_directory)
    filename = output_directory * "/cnot3StepsizeTest_order=$(order)_degree=$(degree)_seed=$(seed)_atol=$(atol)_rtol=$(rtol)_D1=$(D1)_time=$(time)_nthreads=$(nthreads)_usejuqbox=$(use_juqbox)_gradient=$(compute_gradient)_gateDuration=$(Tmax)_nCavityLevels=$(N_osc_levels)_excited=$(start_in_highest_state)"

    if use_juqbox
        @assert order == 2
        @assert degree == 2
        collect_data_juqbox(
            cnot3ret.pcof0, cnot3ret.juqbox_params, cnot3ret.juqbox_wa, time,
            filename, N_timestep_saves, compute_gradient
        )
    else
        if compute_gradient
            collect_data_grad(
                cnot3ret.qgd_prob, controls, cnot3ret.pcof0, cnot3ret.target,
                order, time, filename, N_timestep_saves
            )
        else
            collect_data(
                cnot3ret.qgd_prob, controls, cnot3ret.pcof0, order, time,
                filename, N_timestep_saves
            )
        end
    end
    println(stdout, "Finished! Stored results at:\n\t", filename)

end



function collect_data(prob::SchrodingerProb, controls::ControlsType,
        pcof::AbstractVector{<: Real}, order::Integer, max_walltime::Real,
        filename_base::AbstractString, N_timestep_saves::Integer
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
    prob.nsteps = 2
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


function collect_data_juqbox(pcof0::Vector{Float64}, params::Juqbox.objparams,
        wa::Juqbox.Working_Arrays, max_walltime::Real,
        filename_base::AbstractString, N_timestep_saves::Integer, gradient::Bool=false
    )
    order = 2

    function eval_forward_juqbox(gradient::Bool)
        verbose = true
        evaladjoint = gradient
        returned_tup = traceobjgrad(pcof0, params, wa, verbose, evaladjoint)
        # juqbox_history is ordered differently than QuantumGateDesign
        history = gradient ? returned_tup[3] : returned_tup[2]
        history_reordered = permutedims(history, (1,3,2))
        return history_reordered
    end

    initial_time = time()
    max_walltime *= 60*60 # convert walltime from hours to seconds
    
    filename_csv = filename_base * ".csv"
    filename_final_states_csv = filename_base * "_finalStates.csv"

    header = hcat(
        "nsteps", "stepsize", "elapsed_time", "R_abs_err_L1", "R_abs_err_L2",
        "R_rel_err_L1", "R_rel_err_L2", "R_abs_err_Linf",
    )
    println(stdout, header)

    writedlm(filename_csv, rpad.(header, 24), ',')

    final_states_vec = Vector{ComplexF64}(undef, length(params.Uinit))
    final_states_vec .= NaN
    gmres_tracker = GMRESTracker() # Defaults to NaN values

    # Run simulation
    params.nsteps = 2
    stepsize = params.T / params.nsteps

    # Run simulation once just to get compilation out of the way
    dummy_history = eval_forward_juqbox(gradient)

    is_first_step = true
    history_2h = nothing
    history_h = nothing
    elapsed_time = 0.0

    # Loop until time runs out (with estimator for when we will go overtime on next simulation)
    while (time()-initial_time) < (max_walltime - 2*elapsed_time)
        # Run simulation
        t1 = time()
        history_h = eval_forward_juqbox(gradient)
        t2 = time()
        elapsed_time = t2 - t1

        # Collect data into a row
        if !is_first_step
            R = RichardsonExtrapolation(history_h[:,1:2:end,:], history_2h, order)
            csv_row = hcat(
                params.nsteps, stepsize, elapsed_time, R.abs_err_L1,
                R.abs_err_L2, R.rel_err_L1, R.rel_err_L2, R.abs_err_Linf,
            )
        else
            csv_row = hcat(
                params.nsteps, stepsize, elapsed_time, NaN, NaN, NaN, NaN, NaN,
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
        params.nsteps *= 2
        stepsize = params.T / params.nsteps
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


main()
