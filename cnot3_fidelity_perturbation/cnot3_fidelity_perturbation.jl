using QuantumGateDesign, ArgParse, Random, DelimitedFiles, Distributed, Dates
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

@everywhere using QuantumGateDesign, Random, Dates

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--file_identifier", "-f"
            help = "String to prepend output files with (will be followed by automatically generated fields)."
            arg_type = String
            default = ""
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
        "--output_directory", "-o"
            help = "Directory to store data."
            arg_type = String
            default = "Data"
        "--levels_cavity", "-l"
            help = "Number of energy levels to use for the cavity in the Hamiltonian model."
            arg_type = Int64
            default = 4
        "--gate_duration", "-d"
            help = "Duration of the gate, in nanoseconds."
            arg_type = Float64
            default = 550.0
        "--cost_type", "-c"
            help = "Cost type to use as the primary objective function. Valid options are Infidelity, GeneralizedInfidelity, Tracking, and Norm."
            arg_type = Symbol
            default = :Infidelity
        "order"
            help = "Method order to use"
            required = true
            arg_type = Int64
        "degree"
            help = "Degree of B-spline to use"
            required= true
            arg_type = Int64
        "seed"
            help = "Seed to use when generating control vector."
            required = true
            arg_type = Int64
        "fine_target_error"
            help = "Number of timesteps to use for the 'fine' solves."
            required= true
            arg_type = String
        "coarse_target_error"
            help = "Number of timesteps to use for the 'coarse' solves."
            required= true
            arg_type = String
        "npert"
            help = "Number of perturbations to test."
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
    npert = parsed_args["npert"]
    output_directory = parsed_args["output_directory"]
    file_identifier = parsed_args["file_identifier"]
    cost_type = parsed_args["cost_type"]
    Tmax = parsed_args["gate_duration"]
    N_osc_levels = parsed_args["levels_cavity"]
    coarse_target_error = parsed_args["coarse_target_error"]
    fine_target_error = parsed_args["fine_target_error"]
    nthreads = Threads.nthreads()


    println("Running test with the following arguments:")
    for (arg,val) in parsed_args
        println(rpad(arg, 20), " => ", val)
    end

    nsteps_matrix = [
      7431 825 379 182 175 92
      25323 1944 705 387 297 191
      79198 3535 1152 609 388 286
      243338 6301 1702 821 507 352
      778363 11213 2507 1105 642 432
      2489745 19945 3686 1477 813 530
      7963927 35470 5414 1975 1029 644
    ]
    target_errors=["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]
    orders = [2, 4, 6, 8, 10, 12]

    order_i = findfirst(isequal(order), orders)
    order6_i = findfirst(isequal(6), orders)

    coarse_target_error_i = findfirst(isequal(coarse_target_error), target_errors)
    fine_target_error_i = findfirst(isequal(fine_target_error), target_errors)

    nsteps_fine = nsteps_matrix[fine_target_error_i, order6_i]
    nsteps_coarse = nsteps_matrix[coarse_target_error_i, order_i]

    cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1, N_osc_levels=N_osc_levels, Tmax=Tmax)

    println("Schrodinger Problem:")
    display(cnot3ret.qgd_prob)

    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between amax and -amax
    pcof0 = 2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(seed), N_coeff))

    println("[ ", now(), " | worker ", myid(), " ] ", "Getting fine solution")
    cnot3ret.qgd_prob.nsteps = nsteps_fine
    history_fine = eval_forward(cnot3ret.qgd_prob, controls, pcof0, order=6)
    target = history_fine[:,end,:]

    println("[ ", now(), " | worker ", myid(), " ] ", "Getting first coarse solution")
    cnot3ret.qgd_prob.nsteps = nsteps_coarse
    history_coarse = eval_forward(cnot3ret.qgd_prob, controls, pcof0, order=order)
    UT_coarse = history_coarse[:,end,:]
    real_objective = QuantumGateDesign.cost_function(UT_coarse, target, cnot3ret.qgd_prob.N_ess_levels, cost_type=cost_type)

    
    pert_orders = (1e-1, 1e-2, 1e-3)
    println("[ ", now(), " | worker ", myid(), " ] ", "Getting remaining coarse solutions")
    #data = mapreduce(vcat, 1:npert, pert_orders) do pert_i, pert_order
    data = @distributed (vcat) for (pert_i, pert_order) in collect(Iterators.product(1:npert, pert_orders))

        # Perturbation coefficients uniformly distributed between 0.1amax and -0.1amax
        #pcof_pert = 0.2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(i), N_coeff))
        pcof_pert_dir = 2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(pert_i), N_coeff))
        pcof_coarse = pcof0 + pert_order * pcof_pert_dir
        println("[ ", now(), " | worker ", myid(), " ] ", "Getting coarse solution ", pert_i, ", with pert_order ", pert_order)
        history_coarse = eval_forward(cnot3ret.qgd_prob, controls, pcof_coarse, order=order)
        UT_coarse = history_coarse[:,end,:]
        approx_objective = QuantumGateDesign.cost_function(UT_coarse, target, cnot3ret.qgd_prob.N_ess_levels, cost_type=cost_type)

        abs_err = abs(real_objective-approx_objective)
        rel_err = abs((real_objective-approx_objective)/real_objective)

        println("[ ", now(), " | worker ", myid(), " ] ", "Got coarse solution ", pert_i, ", with pert_order ", pert_order, ", rel_err = ", rel_err)
        
        data_row = hcat(order, pert_order, real_objective, approx_objective,
                        abs_err, rel_err)
    end

    mkpath(output_directory)
    filename = "cnot3PerturbationTest_order=$(order)_degree=$(degree)_seed=$(seed)_targetErrorFine=$(fine_target_error)_targetErrorCoarse=$(coarse_target_error)_nstepsFine=$(nsteps_fine)_nstepsCoarse=$(nsteps_coarse)_atol=$(atol)_rtol=$(rtol)_D1=$(D1)_costType=$(cost_type)_gateDuration=$(Tmax)_nCavityLevels=$(N_osc_levels)"
    if !isempty(file_identifier)
        filename = output_directory * "/" * file_identifier * "_" * filename
    else
        filename = output_directory * "/" * filename
    end

    header = hcat("method_order", "pert_order", "true_objective", 
                  "approx_objective", "error", "relative_error")
    open(filename * ".csv", "w") do io
        DelimitedFiles.writedlm(io, rpad.(header, 24), ',')
    end
    open(filename * ".csv", "a+") do io
        DelimitedFiles.writedlm(io, rpad.(data, 24), ',')
    end

    return nothing
end

main()
