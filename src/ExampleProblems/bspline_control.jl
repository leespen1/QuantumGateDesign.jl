"""
One thing I really want to be careful of is whether 'p' and 'q' are the real
parts of the bspline with carrier wave, or just of the bspline. In the paper,
they are the real parts of the bspline only, but I think in Juqbox (where I
got the bsplines from) they include the carrier wave as well.

Do half the controls affect only p, and the other half affect only q? Or is
there more to it?
"""
function bspline_control(bcpar::bcparams)
    p_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 0, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 0, pcof)
    ]
    q_vec = [
        (t, pcof) -> bcarrier2(t,    bcpar, 1, pcof),
        (t, pcof) -> bcarrier2_dt(t, bcpar, 1, pcof)
    ]
    # Gradients for bcarrier actually don't depend on pcof, since the bcarrier
    # functions are linear in the control coefficients
    grad_p_vec = [
        (t, pcof) -> gradbcarrier2(t, bcpar, 0),
        (t, pcof) -> gradbcarrier2_dt(t, bcpar, 0)
    ]
    grad_q_vec = [
        (t, pcof) -> gradbcarrier2(t, bcpar, 1),
        (t, pcof) -> gradbcarrier2_dt(t, bcpar, 1)
    ]

    return Control(p_vec, q_vec, grad_p_vec, grad_q_vec, bcpar.Ncoeff)
end

# Alternate constructor, provide arguments used to construct bcparams directly.
# (except pcof, which I think shouldn't be a constructor arg anyway)
function bspline_control(T, D1, omega)
    pcof = zeros(2*D1) # For now, only doing one coupled pair of control
    bcpar = bcparams(T, D1, omega, pcof)
    return bspline_control(bcpar)
end

"""
Single qubit in the rotating frame, with rotating wave approximation.
"""
function single_qubit_prob_with_bspline_control(detuning_frequency,
        self_kerr_coefficient, N_ess_levels, N_guard_levels;
        N_coeff_per_control = 6, tf=1.0, nsteps=10, 
    )

    prob = single_transmon_qubit_rwa(
        detuning_frequency, self_kerr_coefficient, N_ess_levels,
        N_guard_levels, tf, nsteps
    )

    omega::Vector{Vector{Float64}} = [[detuning_frequency]] # 1 frequency for 1 pair of coupled controls (p and q)
    control = bspline_control(prob.tf, N_coeff_per_control, omega)

    return prob, control
end
