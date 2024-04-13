function plot_control(control::AbstractControl, pcof::AbstractVector{Float64};
        npoints=1001, derivative_orders=0, convert_units=false, linewidth=2)
    t_grid = LinRange(0, control.tf, npoints)
    p_control_vals = zeros(npoints)
    q_control_vals = zeros(npoints)

    pl_vec = []

    for derivative_order in derivative_orders
        for (i, t) in enumerate(t_grid)
            p_control_vals[i] = eval_p_derivative(control, t, pcof, derivative_order)
            q_control_vals[i] = eval_q_derivative(control, t, pcof, derivative_order)
        end
        if convert_units
            p_control_vals .*= 1e3/(2pi)
            q_control_vals .*= 1e3/(2pi)
        end
        pl = Plots.plot(t_grid, p_control_vals, label="Real", lw=linewidth)
        Plots.plot!(t_grid, q_control_vals, label="Imag", lw=linewidth)
        Plots.plot!(xlabel="Time (ns)", title="Derivative $derivative_order")

        if convert_units
            Plots.plot!(ylabel="Amplitude/2π (MHz)")
        end

        push!(pl_vec, pl)
    end

    return Plots.plot!(pl_vec...)
end

#=
function plot_control(control::HermiteCarrierControl, pcof;
        npoints=1001, derivative_orders=0, convert_units=false, linewidth=2, w=0.0)
    t_grid = LinRange(0, control.tf, npoints)
    p_control_vals = zeros(npoints)
    q_control_vals = zeros(npoints)

    pl_vec = []

    for derivative_order in derivative_orders
        for (i, t) in enumerate(t_grid)
            p_control_vals[i] = eval_derivative(control, t, pcof, derivative_order, :p, w)
            q_control_vals[i] = eval_derivative(control, t, pcof, derivative_order, :q, w)
        end
        if convert_units
            p_control_vals .*= 1e3/(2pi)
            q_control_vals .*= 1e3/(2pi)
        end
        pl = Plots.plot(t_grid, p_control_vals, label="Real", lw=linewidth)
        Plots.plot!(t_grid, q_control_vals, label="Imag", lw=linewidth)
        Plots.plot!(xlabel="Time (ns)", title="Derivative $derivative_order")

        if convert_units
            Plots.plot!(ylabel="Amplitude/2π (MHz)")
        end

        push!(pl_vec, pl)
    end

    return Plots.plot!(pl_vec...)
end
=#
