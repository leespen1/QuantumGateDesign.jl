using Distributed, SlurmClusterManager
using QuantumGateDesign, DelimitedFiles
using QuantumGateDesign: setup_cnot3, get_D1, get_controls

if haskey(ENV, "SLURM_JOB_ID") # Set up remote processes if in SLURM
    println("In SLURM environment, using SlurmClusterManager")
    addprocs(SlurmManager(), exeflags="--project")
else 
    println("Running locally")
    println("Total memory is: ", Sys.total_memory())
    #addprocs(Sys.CPU_THREADS-1)
end

@everywhere begin
    println("[After addprocs] Hello from $(myid()):$(gethostname())\nCurrent project environemtn $(Base.active_project())\nCurrent Directory: $(pwd())")
    using QuantumGateDesign, Dates
    using QuantumGateDesign: real_to_complex, get_number_of_control_parameters,
                             DiscreteAdjointTimes, discrete_adjoint!, total_time,
                             cost_function
    using LinearAlgebra: norm
end

function val_parse(value::AbstractString)
    pvalue = tryparse(Int, value) # parsed value
    pvalue = isnothing(pvalue) ? tryparse(Float64, value) : pvalue
    pvalue = isnothing(pvalue) ? tryparse(Bool, value) : pvalue
    pvalue = isnothing(pvalue) ? string(value) : pvalue
    return pvalue
end

function parse_filename_params(filename::String)
    # Extract key=value pairs
    reduced_filename = first(splitext(basename(filename))) # Remove directory and extension
    key_val_regex = r"([a-zA-Z0-9]+)=([^\_]+)"
    #key_val_regex = r"(\w+)=([^\._]+)" # \w+ also include underscores, hence why I don't use
    matches = eachmatch(key_val_regex, reduced_filename)
    return Dict(m.captures[1] => val_parse(m.captures[2]) for m in matches)
end


function main()
    println("Starting Main")
    filepath = ENV["CNOT3FILEPATH"]
    #filepath = "/mnt/ffs24/home/leespen1/Research/QuantumGateDesign.jl/cnot3_optimization/53220812/targetError=1e-1_cnot3OptimizationTest_order=4_degree=14_seed=0_nsteps=825_atol=1.0e-10_rtol=1.0e-10_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv"
    #filepath = "/mnt/ffs24/home/leespen1/Research/QuantumGateDesign.jl/cnot3_optimization/53220812/targetError=1e-3_cnot3OptimizationTest_order=4_degree=14_seed=0_nsteps=3535_atol=1.0e-10_rtol=1.0e-10_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv"
    filename = basename(filepath)

    parsed_args = parse_filename_params(filename) 
    coarse_order = parsed_args["order"]
    degree = parsed_args["degree"]
    #coarse_atol = parsed_args["atol"]
    #coarse_rtol = parsed_args["rtol"]
    coarse_nsteps = parsed_args["nsteps"]
    D1 = parsed_args["D1"]
    Tmax = parsed_args["gateDuration"]
    N_osc_levels = parsed_args["nCavityLevels"]
    target_error = parsed_args["targetError"]

    coarse_N_deriv = div(coarse_order, 2)

    # Used when aiming for accuracy
    atol = 1e-15
    rtol = 1e-15
    #fine_atol = 1e-15
    #fine_rtol = 1e-15
    fine_order = 6
    fine_nsteps = 5415
    fine_N_deriv = div(fine_order, 2)

    cnot3ret = QuantumGateDesign.setup_cnot3(
        seed=0, # Doesn't matter
        atol=atol,
        rtol=rtol,
        D1=D1,
        N_osc_levels=N_osc_levels,
        Tmax=Tmax
    )
    cnot3ret2 = QuantumGateDesign.setup_cnot3(
        seed=0, # Doesn't matter
        atol=atol,
        rtol=rtol,
        D1=D1,
        N_osc_levels=N_osc_levels,
        Tmax=Tmax
    )
    coarse_prob = cnot3ret.qgd_prob
    coarse_prob.nsteps = coarse_nsteps
    fine_prob = cnot3ret2.qgd_prob
    fine_prob.nsteps = fine_nsteps
    target = cnot3ret.target

    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    header = hcat(
        "ipopt_iter",
        "UT_err", "grad_pcof_err", "infidelity_err", "gen_infidelity_err",
        "coarse_infidelity", "coarse_gen_infidelity", "coarse_norm_grad", "coarse_norm_UT",
        "fine_infidelity", "fine_gen_infidelity", "fine_norm_grad", "fine_norm_UT",
    )

    pcofs = readdlm(filepath, ',', Float64)
    #pcofs = pcofs[1:500, :] # Don't do the whole iteration history, too much work
    pcof_rows = copy.(eachrow(pcofs)) # Copy to convert to plain arrays, not views
    chunk_size = 10*nworkers()
    pcof_rows_chunked = copy.(collect.(Iterators.partition(pcof_rows, chunk_size)))

    out_filename = "comparison_" * basename(filename)
    open(out_filename, "w") do io
        DelimitedFiles.writedlm(io, rpad.(header, 24), ',')
    end

    for (i_chunk, pcof_rows_chunk) in enumerate(pcof_rows_chunked)
        println("[ $(now()) | worker  $(myid()) ] Starting chunk $i_chunk")

        data = @distributed (vcat) for (iter_n, pcof) in collect(enumerate(pcof_rows_chunk))
            ipopt_iter = ((i_chunk-1)*chunk_size+iter_n)
            println("[ $(now()) | worker  $(myid()) ] Doing iteration $ipopt_iter")

            # Allocate working arrays
            coarse_history = zeros(coarse_prob.real_system_size, 1+coarse_N_deriv, 1+coarse_nsteps, coarse_prob.N_initial_conditions)
            coarse_lambda_history = copy(coarse_history) 
            coarse_adjoint_forcing = zeros(coarse_prob.real_system_size, 1+coarse_nsteps, coarse_prob.N_initial_conditions)
            coarse_grad = zeros(N_coeff)

            # Compute gradient
            discrete_adjoint!(
                coarse_grad, coarse_history, coarse_lambda_history, coarse_adjoint_forcing,
                coarse_prob, controls, pcof, target, order=coarse_order
            )

            # Compute data values for coarse history
            coarse_UT = real_to_complex(coarse_history[:,1,end,:])
            coarse_inf = cost_function(coarse_UT, target, coarse_prob.N_ess_levels,
                                       cost_type=:Infidelity)
            coarse_ginf = cost_function(coarse_UT, target, coarse_prob.N_ess_levels,
                                       cost_type=:GeneralizedInfidelity)
            coarse_norm_grad = norm(coarse_grad)
            coarse_norm_UT = norm(coarse_UT)

            fine_history = zeros(fine_prob.real_system_size, 1+fine_N_deriv, 1+fine_nsteps, fine_prob.N_initial_conditions)
            fine_lambda_history = copy(fine_history) 
            fine_adjoint_forcing = zeros(fine_prob.real_system_size, 1+fine_nsteps, fine_prob.N_initial_conditions)
            fine_grad = zeros(N_coeff)

            # Compute gradient
            discrete_adjoint!(
                fine_grad, fine_history, fine_lambda_history, fine_adjoint_forcing,
                fine_prob, controls, pcof, target, order=fine_order
            )

            # Compute data values for fine history
            fine_UT = real_to_complex(fine_history[:,1,end,:])
            fine_inf = cost_function(fine_UT, target, fine_prob.N_ess_levels,
                                       cost_type=:Infidelity)
            fine_ginf = cost_function(fine_UT, target, fine_prob.N_ess_levels,
                                       cost_type=:GeneralizedInfidelity)
            fine_norm_grad = norm(fine_grad)
            fine_norm_UT = norm(fine_UT)

            # Compute remaining data values (which depend on coarse and fine)
            UT_err = norm(coarse_UT - fine_UT)
            grad_pcof_err = norm(coarse_grad - fine_grad)
            inf_err = abs(coarse_inf - fine_inf)
            ginf_err = abs(coarse_ginf - fine_ginf)

            data_row = hcat(
                ipopt_iter, UT_err, grad_pcof_err, inf_err, ginf_err, 
                coarse_inf, coarse_ginf, coarse_norm_grad, coarse_norm_UT,
                fine_inf, fine_ginf, fine_norm_grad, fine_norm_UT,
            )
        end

        open(out_filename, "a+") do io
            DelimitedFiles.writedlm(io, rpad.(data, 24), ',')
        end
    end
    
    return nothing
end

main()
