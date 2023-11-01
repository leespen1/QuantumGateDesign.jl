"""
One thing I really want to be careful of is whether 'p' and 'q' are the real
parts of the bspline with carrier wave, or just of the bspline. In the paper,
they are the real parts of the bspline only, but I think in Juqbox (where I
got the bsplines from) they include the carrier wave as well.
"""
function bspline_control(bcpar)
    p_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 0, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 0, pcof)
    ]
    q_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 1, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 1, pcof)
    ]
    # Gradients for bcarrier actually don't depend on pcof.
    grad_p_vec = [
        (t, pcof) -> gradbcarrier2(t, bcpar, 0),
        (t, pcof) -> gradbcarrier2_dt(t, bcpar, 0)
    ]
    grad_q_vec = [
        (t, pcof) -> gradbcarrier2(t, bcpar, 1),
        (t, pcof) -> gradbcarrier2_dt(t, bcpar, 1)
    ]

    N_coeff = length(bcpar.pcof)
    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, N_coeff)
end

"""
Single qubit in the rotating frame, with rotating wave approximation.
"""
function single_qubit_prob_with_bspline_control(detuning_frequency,
        self_kerr_coefficient, N_ess_levels, N_guard_levels;
        N_coeff_per_control = 4, tf=1.0, nsteps=10, 
    )

    prob = single_transmon_qubit_rwa(
        detuning_frequency, self_kerr_coefficient, N_ess_levels,
        N_guard_levels, tf, nsteps
    )

    omega::Vector{Vector{Float64}} = [[detuning_frequency]] # 1 frequency for 1 pair of coupled controls (p and q)
    pcof = ones(2*N_coeff_per_control) # Put in dummy pcof, just so the length is known
    # Use simplest constructor
    bcpar = bcparams(prob.tf, N_coeff_per_control, omega, pcof) 

    control = bspline_control(bcpar)

    return prob, control
end
