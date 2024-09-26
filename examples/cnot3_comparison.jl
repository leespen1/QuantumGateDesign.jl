include("cnot3_setup.jl")
using JLD2, Dates

prob.gmres_abstol = 1e-12
prob.gmres_reltol = 1e-12
run_convergence = false
if run_convergence
    N_iter = 20
    base_nsteps = 10
    min_error_limit = 1e-12
    max_error_limit = 5e-7

    ret_qgd = get_histories(
        prob, controls, pcof0, N_iter,
        base_nsteps=base_nsteps, min_error_limit=min_error_limit,
        max_error_limit=max_error_limit
    )

    ret_juq = get_histories(params, wa, pcof0, N_iter, base_nsteps=base_nsteps)
    ret_full = merge(ret_qgd, ret_juq)

    pl1 = QuantumGateDesign.plot_stepsize_convergence(ret_full)
    pl2 = QuantumGateDesign.plot_timing_convergence(ret_full)

    jldsave("results_cnot3_comparison_" * string(now()) * ".jld2";
            ret_full,
            prob.gmres_abstol,
            prob.gmres_reltol,
            N_iter,
            base_nsteps,
            min_error_limit,
            prob,
            params,
            wa,
            pcof0
    )
end

#juqbox_ipopt_prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=maxIter, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)
run_optimiation = false
if run_optimiation
    prob.nsteps = 5000
    opt_ret = optimize_gate(
        prob, controls, pcof0, target, order=6,
        pcof_U=amax, pcof_L=-amax,
        maxIter=10_000, max_cpu_time=60.0*60*2
    )
    #pcof = Juqbox.run_optimizer(juqbox_ipopt_prob, pcof0);
end
