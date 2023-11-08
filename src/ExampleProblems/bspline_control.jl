"""
One thing I really want to be careful of is whether 'p' and 'q' are the real
parts of the bspline with carrier wave, or just of the bspline. In the paper,
they are the real parts of the bspline only, but I think in Juqbox (where I
got the bsplines from) they include the carrier wave as well.

Do half the controls affect only p, and the other half affect only q? Or is
there more to it?
"""
#=
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
#
# Used for a single coupled control with multiple frequencies
function bspline_control(T::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(T, D1, omega_bcpar, pcof)
    return bspline_control(bcpar)
end
=#
