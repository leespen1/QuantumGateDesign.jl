using QuantumGateDesign, DelimitedFiles, JLD2
using Juqbox, JLD2, Dates, Printf, Random, LinearAlgebra, Pkg, InteractiveUtils, ArgParse
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

function collect_data(prob::SchrodingerProb, controls,
        pcof::AbstractVector{<: Real}, order::Integer, max_walltime::Real,
        filename_base::AbstractString
    )
    max_walltime *= 60 # convert walltime from minutes to seconds

    initial_time = time()
    prob = copy(prob) # Copy problem, just to make sure there are no mutability issues.

    filename_jld2 = filename_base * ".jld2"
    filename_csv = filename_base * ".csv"
    csv_data::Matrix{Any} = ["nsteps" "stepsize" "R_abs_err_L1" "R_abs_err_L2" "R_rel_err_L1" "R_rel_err_L2" "R_abs_err_Linf" "elapsed_time"]

    # Run simulation
    prob.nsteps = 2
    stepsize = prob.tf / prob.nsteps
    t1 = time()
    history_2h = eval_forward(prob, controls, pcof, order=order)
    t2 = time()
    elapsed_time = t2 - t1
    # Log data (CSV)
    csv_row = transpose([prob.nsteps, stepsize, NaN, NaN, NaN, NaN, NaN, elapsed_time])
    csv_data = [csv_data; csv_row]
    writedlm(filename_csv, csv_data, ',')
    println(csv_row)
    # Log data (JLD2)
    jldopen(filename_jld2, "w") do file
        file[string(prob.nsteps)] = history_2h
    end

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

        # Log data (CSV)
        R = QuantumGateDesign.RichardsonExtrapolation(history_h[:,1:2:end,:], history_2h, order)
        csv_row = [prob.nsteps stepsize R.abs_err_L1 R.abs_err_L2 R.rel_err_L1 R.rel_err_L2 R.abs_err_Linf elapsed_time]
        csv_data = [csv_data; csv_row]
        writedlm(filename_csv, csv_data, ',')
        println(csv_row)
        println("Size of csv_data:\t", sizeof(csv_data))
        println("Size of history_h:\t", sizeof(history_h))
        println("Size of free memory:\t", Int(Sys.free_memory()))

        # Log data (JLD2)
        jldopen(filename_jld2, "a+") do file
            file[string(prob.nsteps)] = history_h
        end

        history_2h = history_h
    end

    return readdlm(filename_csv, ',')
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
            help = "Amount of wall time (in minutes) to spend on test."
            arg_type = Float64
            default = 1 # Default
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
    nthreads = Threads.nthreads()
    cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, rtol=rtol, D1=D1)
    controls = get_controls(order, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)
    # Coefficients uniformly distributed between amax and -amax
    pcof = cnot3ret.amax * 2* (0.5 .- rand(MersenneTwister(seed), N_coeff))

    mkdir("Data")
    filename = "Data/cnot3StepsizeTest_seed=$(seed)_order=$(order)_rtol=$(rtol)_D1=$(D1)_time=$(time)_nthreads=$(nthreads)"

    collect_data(cnot3ret.qgd_prob, controls, cnot3ret.pcof0, order, time, filename)
end

main()
