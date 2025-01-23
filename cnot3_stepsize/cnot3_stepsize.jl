using QuantumGateDesign, DelimitedFiles, JLD2
using Juqbox, Dates, Printf, Random, LinearAlgebra, Pkg, InteractiveUtils, ArgParse
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

function collect_data(prob::SchrodingerProb, controls,
        pcof::AbstractVector{<: Real}, order::Integer, max_walltime::Real,
        filename_base::AbstractString, N_timestep_saves::Integer
    )
    prob = copy(prob) # Copy problem, just to make sure there are no mutability issues.

    initial_time = time()
    max_walltime *= 60*60 # convert walltime from hours to seconds
    

    filename_csv = filename_base * ".csv"
    filename_final_states_csv = filename_base * "_finalStates.csv"
    filename_state_saves_csv = filename_base * "_Nsaves=$(N_timestep_saves).csv"

    csv_data::Matrix{Any} = ["nsteps" "stepsize" "R_abs_err_L1" "R_abs_err_L2" "R_rel_err_L1" "R_rel_err_L2" "R_abs_err_Linf" "elapsed_time"]
    final_states = Matrix{ComplexF64}(undef, 0, prob.N_tot_levels*prob.N_initial_conditions)
    state_saves = Matrix{ComplexF64}(undef, 0, prob.N_tot_levels*prob.N_initial_conditions*(1+N_timestep_saves))

    # Run simulation
    prob.nsteps = 2
    stepsize = prob.tf / prob.nsteps
    t1 = time()
    history_2h = eval_forward(prob, controls, pcof, order=order)
    t2 = time()
    elapsed_time = t2 - t1
    # Store/process data
    final_states = [final_states; reshape(history_2h[:,end,:], 1, :)]
    state_saves = [state_saves; parse_history_for_csv(history_2h, N_timestep_saves)]

    csv_row = transpose([prob.nsteps, stepsize, NaN, NaN, NaN, NaN, NaN, elapsed_time])
    csv_data = [csv_data; csv_row]

    # Log data (CSV)
    writedlm(filename_csv, csv_data, ',')
    writedlm(filename_final_states_csv, final_states, ',')
    writedlm(filename_state_saves_csv, state_saves, ',')

    # Loop until time runs out (with estimator for when we will go overtime on next simulation)
    while (time()-initial_time) < (max_walltime - 2*elapsed_time)
        # Set new problem parameters
        prob.nsteps *= 2
        stepsize = prob.tf / prob.nsteps

        # Check that we have enough memory
        estimated_memory = sizeof(history_2h)*div(order,2)*2
        free_memory = Sys.free_memory()
        if  estimated_memory > free_memory
            @warn "Ending early because iteration with $(prob.nsteps) steps is estimated to use $estimated_memory bytes of memory, but only $free_memory bytes of RAM remain."
            break
        end

        # Run simulation
        t1 = time()
        history_h = eval_forward(prob, controls, pcof, order=order)
        t2 = time()
        elapsed_time = t2 - t1

        # Store/process data
        final_states = [final_states; reshape(history_h[:,end,:], 1, :)]
        state_saves = [state_saves; parse_history_for_csv(history_2h, N_timestep_saves)]

        R = QuantumGateDesign.RichardsonExtrapolation(history_h[:,1:2:end,:], history_2h, order)
        csv_row = [prob.nsteps stepsize R.abs_err_L1 R.abs_err_L2 R.rel_err_L1 R.rel_err_L2 R.abs_err_Linf elapsed_time]
        csv_data = [csv_data; csv_row]

        # Log data (CSV)
        writedlm(filename_csv, csv_data, ',')
        writedlm(filename_final_states_csv, final_states, ',')
        writedlm(filename_state_saves_csv, state_saves, ',')

        println("Size of csv_data:\t", sizeof(csv_data))
        println("Size of history_h:\t", sizeof(history_h))
        println("Size of free memory:\t", Int(Sys.free_memory()))


        history_2h = history_h
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

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
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
        "order"
            help = "Method order to use"
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
    seed = parsed_args["seed"]
    rtol = parsed_args["rtol"]
    D1 = parsed_args["D1"]
    time = parsed_args["time"]
    N_timestep_saves = parsed_args["nsaves"]
    output_directory = parsed_args["output_directory"]

    nthreads = Threads.nthreads()
    cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, rtol=rtol, D1=D1)
    controls = get_controls(order, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between amax and -amax
    pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))

    mkpath(output_directory)
    filename = output_directory * "/cnot3StepsizeTest_seed=$(seed)_order=$(order)_rtol=$(rtol)_D1=$(D1)_time=$(time)_nthreads=$(nthreads)"

    collect_data(cnot3ret.qgd_prob, controls, cnot3ret.pcof0, order, time, filename, N_timestep_saves)
end

main()
