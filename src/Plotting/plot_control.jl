function plot_control(control::AbstractControl, pcof::AbstractVector{Float64}; npoints=1001, derivative_order=0)
    t_grid = LinRange(0, control.tf, npoints)
    p_control_vals = zeros(npoints)
    q_control_vals = zeros(npoints)
    for (i, t) in enumerate(t_grid)
        p_control_vals[i] = eval_p_derivative(control, t, pcof, derivative_order)
        q_control_vals[i] = eval_q_derivative(control, t, pcof, derivative_order)
    end

    pl = Plots.plot(t_grid, p_control_vals, label="p(t) [d$derivative_order]")
    Plots.plot!(t_grid, q_control_vals, label="q(t) [d$derivative_order]")
    Plots.plot!(ylabel="Amplitude", xlabel="t")

    return pl
end
