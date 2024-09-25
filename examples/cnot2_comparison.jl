include("cnot2_setup.jl")

run_convergence = true
if run_convergence
    N_iter = 15
    base_nsteps = 10
    min_error_limit = 5e-9

    ret_qgd = get_histories(
        prob, controls, pcof0, N_iter,
        base_nsteps=base_nsteps, min_error_limit=min_error_limit
    )

    ret_juq = get_histories(params, wa, pcof0, N_iter, base_nsteps=base_nsteps)
    ret_full = merge(ret_qgd, ret_juq)

    pl1 = QuantumGateDesign.plot_stepsize_convergence(ret_full)
    pl2 = QuantumGateDesign.plot_timing_convergence(ret_full)
end
