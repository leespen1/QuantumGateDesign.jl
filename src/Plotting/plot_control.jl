function plot_control(control::AbstractControl, pcof::AbstractVector{Float64};
        npoints=1001, derivative_order=0, convert_units=false, linewidth=2)
    t_grid = LinRange(0, control.tf, npoints)
    p_control_vals = zeros(npoints)
    q_control_vals = zeros(npoints)
    for (i, t) in enumerate(t_grid)
        p_control_vals[i] = eval_p_derivative(control, t, pcof, derivative_order)
        q_control_vals[i] = eval_q_derivative(control, t, pcof, derivative_order)
    end
    if convert_units
        p_control_vals .*= 1e3/(2pi)
        q_control_vals .*= 1e3/(2pi)
    end
    pl = Plots.plot(t_grid, p_control_vals, label="p(t)", lw=linewidth)
    Plots.plot!(t_grid, q_control_vals, label="q(t)", lw=linewidth)
    Plots.plot!(xlabel="Time (ns)")

    if convert_units
        Plots.plot!(ylabel="Amplitude/2π (MHz)")
    end

    return pl
end
