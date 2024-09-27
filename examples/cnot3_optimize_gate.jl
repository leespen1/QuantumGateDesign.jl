using JLD2, Dates, ArgParse, Printf

s = ArgParseSettings()

@add_arg_table s begin
    "--use_juqbox", "-j"
        help = "Flag for using Juqbox"
        action = :store_true
    "--gmres_abstol", "-a"
        help = "Absolute tolerance to use in GMRES"
        arg_type = Float64
        default = 1e-12
    "--gmres_reltol", "-r"
        help = "Relative tolerance to use in GMRES"
        arg_type = Float64
        default = 1e-12
    "--max_iter", "-m"
        help = "Maximum number of IPOPT iterations."
        arg_type = Int
        default = 50
    "order"
        help = "Order of the method."
        arg_type = Int
        required = true
    "stepsize"
        help = "Stepsize to use."
        arg_type = Float64
        required = true
end

parsed_args = parse_args(s)

order = parsed_args["order"]
stepsize = parsed_args["stepsize"]
use_juqbox = parsed_args["use_juqbox"]
max_iter = parsed_args["max_iter"]

if use_juqbox
    @assert order == 2
end

# Run setup
include("cnot3_setup.jl")

# Compute nsteps based on command line input
nsteps = ceil(Int, prob.tf / stepsize)

prob.nsteps = nsteps
params.nsteps = nsteps

prob.gmres_abstol = parsed_args["gmres_abstol"]
prob.gmres_reltol = parsed_args["gmres_reltol"]


if use_juqbox
    # Run test
    juqbox_ipopt_prob = Juqbox.setup_ipopt_problem(
        params, wa, nCoeff, minCoeff, maxCoeff,
        maxIter=max_iter,
        lbfgsMax=lbfgsMax,
        startFromScratch=startFromScratch,
        max_cpu_time=60.0*60*8
    )
    pcof_opt = Juqbox.run_optimizer(juqbox_ipopt_prob, pcof0);


    jldsave("results_cnot3_juqbox_order$(order)_stepsize" * @sprintf("%.2E", stepsize) * "_"* string(now()) * ".jld2";
        order,
        nsteps,
        use_juqbox,
        pcof_opt,
        max_iter,
        params,
        juqbox_ipopt_prob
    )
else
    opt_ret = optimize_gate(
        prob, controls, pcof0, target, order=order,
        pcof_U=amax, pcof_L=-amax,
        maxIter=max_iter, max_cpu_time=60.0*60*8
    )

    jldsave("results_cnot3_order$(order)_stepsize" * @sprintf("%.2E", stepsize) * "_"* string(now()) * ".jld2";
        order,
        nsteps,
        use_juqbox,
        max_iter,
        prob,
        opt_ret
    )
end
