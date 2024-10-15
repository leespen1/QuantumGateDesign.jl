include("cnot2_setup.jl")

#pcof0_bspline_controls = amax*0.01*rand(get_number_of_control_parameters(bspline_controls))
#pcof0_hermite = amax*0.003*rand(get_number_of_control_parameters(hermite_controls))

function find_true_history(ret_full)
    min_richardson_error = Inf
    min_error_order_str = ""
    min_error_index = -1
    for (order_str, dict) in ret_full
        for (i, error) in enumerate(dict["richardson_errors"])
            if error < min_richardson_error
                min_richardson_error = error
                min_error_order_str = order_str
                min_error_index = i
            end
        end
    end
    true_history = ret_full[min_error_order_str]["histories"][min_error_index]
end



run_convergence = false
if run_convergence
    N_iter = 13
    base_nsteps = 10
    min_error_limit = 1e-10
    prob.gmres_abstol=1e-15
    prob.gmres_reltol=1e-15

    ret_qgd = get_histories(
        prob, bspline_controls, pcof0, N_iter, orders=(2,4),
        base_nsteps=base_nsteps, min_error_limit=min_error_limit
    )

    ret_juq = get_histories(params, wa, pcof0, N_iter, base_nsteps=base_nsteps)
    ret_full = merge(ret_qgd, ret_juq)

    true_history = find_true_history(ret_full)

    #pl1 = QuantumGateDesign.plot_stepsize_convergence(ret_full)
    #pl2 = QuantumGateDesign.plot_timing_convergence(ret_full)
    pl1 = QuantumGateDesign.plot_stepsize_convergence(ret_full, true_history=true_history)
    pl2 = QuantumGateDesign.plot_timing_convergence(ret_full, true_history=true_history)
end

# A major issue I have is controlling the maximum amplitude for hermite controls

run_optimization1 = false
if run_optimization1
    params.nsteps = 400 # Approximately 1e-4 Error for Juqbox
    juqbox_ipopt_prob = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=200, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)
    pcof_opt = Juqbox.run_optimizer(juqbox_ipopt_prob, pcof0);
end

prob.nsteps = 1500
infidelity_after_optimization1 = infidelity(prob, bspline_controls, pcof_opt, target, order=4)

run_optimization2 = true
if run_optimization2
    prob.nsteps = 1500 # Approximately 1e-7 Error for Order 4 Hermite
    opt_ret = optimize_gate(
        prob, bspline_controls, pcof_opt, target, order=4,
        pcof_U=amax, pcof_L=-amax,
        maxIter=50, max_cpu_time=60.0*60*2
    )
end
prob.nsteps = 3000
infidelity_after_optimization2 = infidelity(prob, bspline_controls, opt_ret["optimal_pcof"], target, order=4)

run_optimization3 = true
if run_optimization3
    params.nsteps = 50_000 # Approximately 1e-7 Error for Juqbox
    juqbox_ipopt_prob2 = Juqbox.setup_ipopt_problem(params, wa, nCoeff, minCoeff, maxCoeff, maxIter=50, lbfgsMax=lbfgsMax, startFromScratch=startFromScratch)
    pcof_opt2 = Juqbox.run_optimizer(juqbox_ipopt_prob2, pcof_opt);
end
prob.nsteps = 3000
infidelity_after_optimization3 = infidelity(prob, bspline_controls, pcof_opt2, target, order=4)

println("Infidelity after first Juqbox optimization: ", infidelity_after_optimization1)
println("Infidelity after QGD optimization: ", infidelity_after_optimization2)
println("Infidelity after second Juqbox optimization: ", infidelity_after_optimization3)
