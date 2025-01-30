using QuantumGateDesign, ArgParse, Random
using QuantumGateDesign: setup_cnot3, get_D1, get_controls
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
        "--time", "-t"
            help = "Amount of wall time (in hours) to spend on test."
            arg_type = Float64
            default = 1 # Default
        "--output_directory", "-o"
            help = "Directory to store data."
            arg_type = String
            default = "Data"
        "--maxiter", "-m"
            help = "Maximum number of iterations to perform in IPOPT optimization."
            arg_type = Int64
            default = 10_000
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
        "nsteps"
            help = "Number of timesteps to use."
            required= true
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
    nsteps = parsed_args["nsteps"]
    maxiter = parsed_args["maxiter"]
    output_directory = parsed_args["output_directory"]
    file_identifier = parsed_args["file_identifier"]

    nthreads = Threads.nthreads()
    cnot3ret = QuantumGateDesign.setup_cnot3(seed=seed, atol=atol, rtol=rtol, D1=D1)
    cnot3ret.qgd_prob.nsteps = nsteps
    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between 0.1amax and -0.1amax
    pcof = 0.2 * cnot3ret.amax * (0.5 .- rand(MersenneTwister(seed), N_coeff))

    mkpath(output_directory)
    filename = "cnot3OptimizationTest_order=$(order)_degree=$(degree)_seed=$(seed)_nsteps=$(nsteps)_atol=$(atol)_rtol=$(rtol)_D1=$(D1)_time=$(time)_maxiter=$(maxiter)_nthreads=$(nthreads)"
    if !isempty(file_identifier)
        filename = output_directory * "/" * file_identifier * "_" * filename
    else
        filename = output_directory * "/" * filename
    end

    ipopt_options = (
        "max_iter" => maxiter,
        "max_wall_time" => 60.0*60*time,
        "derivative_test" => "first-order",
        "limited_memory_max_history" => 50,
        "output_file" => filename * ".txt"
    )

    optimization_history = optimize_gate(
        cnot3ret.qgd_prob, controls, pcof, cnot3ret.target, order=order,
        pcof_ubound=cnot3ret.amax, pcof_lbound=-cnot3ret.amax,
        savename=filename,
        ipopt_options = ipopt_options
    )

    return optimization_history
end

main()
