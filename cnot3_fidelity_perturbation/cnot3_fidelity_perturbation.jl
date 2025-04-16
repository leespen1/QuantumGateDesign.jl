using Distributed, SlurmClusterManager
using QuantumGateDesign, ArgParse, Random, DelimitedFiles, Dates, LinearAlgebra
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

if haskey(ENV, "SLURM_JOB_ID") # Set up remote processes if in SLURM
    println("In SLURM environment, using SlurmClusterManager")
    addprocs(SlurmManager(), exeflags="--project")
else 
    println("Running locally")
    println("Total memory is: ", Sys.total_memory())
    #addprocs(Sys.CPU_THREADS-1)
end

println("Workers:", workers())

@everywhere begin
    println("[After addprocs] Hello from $(myid()):$(gethostname())\nCurrent project environemtn $(Base.active_project())\nCurrent Directory: $(pwd())")
    using QuantumGateDesign, Random, Dates
    using QuantumGateDesign: real_to_complex, get_number_of_control_parameters,
                             DiscreteAdjointTimes, discrete_adjoint!, total_time,
                             cost_function
    using LinearAlgebra: norm
end



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
    println("STARTING AT TIME $(now())")
    parsed_args = parse_commandline()
    order = parsed_args["order"]
    degree = 14
    seed = parsed_args["seed"]
    atol = parsed_args["atol"]
    rtol = parsed_args["rtol"]
    D1 = 16
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

    cnot3ret = setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1, N_osc_levels=N_osc_levels, Tmax=Tmax)
    
    prob = cnot3ret.qgd_prob


    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)
    N_coeff = get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between amax and -amax
    #pcof0 = 2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(seed), N_coeff))
    input_pcof_str = "targetError=1e-7_cnot3OptimizationTest_order=6_degree=14_seed=0_nsteps=5414_atol=1.0e-15_rtol=1.0e-15_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=Infidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv"
    pcofs, header = readdlm(input_pcof_str, ',', Float64, header=true)
    pcof0 = pcofs[rand(50:end),:] # Use a random control vector that is a little bit in the middle of the optimization
    #pcof0_avg = norm(pcof0, 1) / length(pcof0)


    println("[ ", now(), " | worker ", myid(), " ] ", "Getting fine solution\n")
    prob.nsteps = nsteps_fine
    history_fine = eval_forward(prob, controls, pcof0, order=6)
    target = history_fine[:,end,:]

    println("[ ", now(), " | worker ", myid(), " ] ", "Getting first coarse solution\n")

    prob.nsteps = nsteps_coarse
    N_derivatives = div(order, 2) 
    real_grad = zeros(get_number_of_control_parameters(controls))
    history_coarse0 = zeros(prob.real_system_size, 1+N_derivatives, 1+nsteps_coarse, prob.N_initial_conditions)
    lambda_history_coarse0 = zeros(prob.real_system_size, 1+N_derivatives, 1+nsteps_coarse, prob.N_initial_conditions)
    adjoint_forcing_coarse0 = zeros(prob.real_system_size, 1+nsteps_coarse, prob.N_initial_conditions)
    
    timer = DiscreteAdjointTimes()
    forward_gmres_tracker = GMRESTracker()
    adjoint_gmres_tracker = GMRESTracker()

    discrete_adjoint!(
        real_grad, history_coarse0, lambda_history_coarse0, adjoint_forcing_coarse0, prob,
        controls, pcof0, target, order=order, timer=timer,
        forward_gmres_tracker=forward_gmres_tracker,
        adjoint_gmres_tracker=adjoint_gmres_tracker,
    )
    UT_coarse0 = real_to_complex(history_coarse0[:,1,end,:])

    real_objective = cost_function(UT_coarse0, target, prob.N_ess_levels, cost_type=cost_type)
    real_grad_norm = norm(real_grad)
    real_grad_norm_inf = norm(real_grad, Inf)


    pert_orders = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
    #data = mapreduce(vcat, 1:npert, pert_orders) do pert_i, pert_order
    data = @distributed (vcat) for (pert_i, pert_order) in collect(Iterators.product(1:npert, pert_orders))
        println("[ $(now()) | worker  $(myid()) ] Getting coarse solution $pert_i with pert_order $pert_order")

        # Perturbation coefficients uniformly distributed between 0.1amax and -0.1amax
        #pcof_pert = 0.2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(i), N_coeff))
        pcof_pert_dir = 2 * (0.5 .- rand(MersenneTwister(pert_i), N_coeff))
        pcof_coarse = pcof0 + pert_order * pcof_pert_dir

        # TODO add gradient norm, final state, comparison 
        pert_history = zeros(prob.real_system_size, 1+N_derivatives, 1+nsteps_coarse, prob.N_initial_conditions)
        pert_lambda_history = zeros(prob.real_system_size, 1+N_derivatives, 1+nsteps_coarse, prob.N_initial_conditions)
        pert_adjoint_forcing = zeros(prob.real_system_size, 1+nsteps_coarse, prob.N_initial_conditions)
        pert_grad = zeros(get_number_of_control_parameters(controls))

        pert_timer = DiscreteAdjointTimes()
        pert_forward_gmres_tracker = GMRESTracker()
        pert_adjoint_gmres_tracker = GMRESTracker()

        discrete_adjoint!(
            pert_grad, pert_history, pert_lambda_history, pert_adjoint_forcing,
            prob, controls, pcof, target, order=order, timer=pert_timer,
            forward_gmres_tracker=pert_forward_gmres_tracker,
            adjoint_gmres_tracker=pert_adjoint_gmres_tracker,
        )

        pert_UT = real_to_complex(pert_history[:,1,end,:])

        pert_objective = cost_function(
            pert_UT, target, prob.N_ess_levels,
            cost_type=cost_type
        )
        obj_err = abs(real_objective-pert_objective)

        pert_grad_norm = norm(pert_grad)
        pert_grad_norm_inf = norm(pert_grad, Inf)

        grad_err = norm(pert_grad - real_grad)
        grad_err_inf = norm(pert_grad - real_grad, Inf)

        UT_coarse_err = norm(pert_UT - UT_coarse)
        UT_fine_err = norm(pert_UT - UT_fine)

        avg_gmres_iter = avg_N_iterations(gmres_tracker)

        println("[ ", now(), " | worker ", myid(), " ] ", "Got coarse solution ", pert_i, ", with pert_order ", pert_order, ", rel_err = ", rel_err)
        
        data_row = hcat(
            order, pert_i, pert_order, real_objective, pert_objective, obj_err,
            real_grad_norm, pert_grad_norm, grad_err, 
            real_grad_norm_inf, pert_grad_norm_inf, grad_err_inf, 
            UT_coarse_err, UT_fine_err, avg_gmres_iter
        )
    end

    mkpath(output_directory)
    filename = "cnot3PerturbationTest_order=$(order)_degree=$(degree)_seed=$(seed)_targetErrorFine=$(fine_target_error)_targetErrorCoarse=$(coarse_target_error)_nstepsFine=$(nsteps_fine)_nstepsCoarse=$(nsteps_coarse)_atol=$(atol)_rtol=$(rtol)_D1=$(D1)_costType=$(cost_type)_gateDuration=$(Tmax)_nCavityLevels=$(N_osc_levels)"
    if !isempty(file_identifier)
        filename = output_directory * "/" * file_identifier * "_" * filename
    else
        filename = output_directory * "/" * filename
    end

    header = hcat(
        "method_order", "pert_i", "pert_order", "real_objective", 
        "pert_objective", "real_grad_norm", "pert_grad_norm", "grad_err_norm",
        "real_grad_norm_inf", "pert_grad_norm_inf", "grad_err_norm_inf",
        "UT_coarse_err", "UT_fine_err", "avg_N_gmres_iter"
    )
    open(filename * ".csv", "w") do io
        DelimitedFiles.writedlm(io, rpad.(header, 24), ',')
    end
    open(filename * ".csv", "a+") do io
        DelimitedFiles.writedlm(io, rpad.(data, 24), ',')
    end

    println("ENDING AT TIME $(now())")
    return nothing
end

main()
