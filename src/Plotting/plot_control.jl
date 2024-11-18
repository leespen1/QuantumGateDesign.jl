"""
    plot_controls(controls, pcof; [npoints=1001, derivative_orders=0, convert_units=false, linewidth=2])

Plot the pulse amplitudes over time for the given control and control vector.
Return the plot.

The derivatives can also be plotted by supplying an integer or vector of integers as the arguemnt for `derivative_orders`.
"""
function plot_controls(controls::Union{AbstractControl, Vector{<: AbstractControl}},
        pcof::AbstractVector{<: Real};
        npoints=201, derivative_orders=0, convert_units=false, linewidth=2,
        control_indices=1:length(controls))

    t_grid = LinRange(0, controls[1].tf, npoints)
    p_control_vals = zeros(npoints, length(control_indices))
    q_control_vals = similar(p_control_vals)

    colors = [Plots.palette(:auto)[k] for k in control_indices]
    colors_mat = reshape(colors, 1, :) # Plots expects a row matrix

    labels = ["Control $k" for k in control_indices]
    labels_mat = reshape(labels, 1, :)

    pl_vec = []

    for derivative_order in derivative_orders
        for (i, t) in enumerate(t_grid)
            for (matrix_col_index, control_index) in enumerate(control_indices)
                control = controls[control_index]
                local_pcof = get_control_vector_slice(pcof, controls, control_index)
                p_control_vals[i,matrix_col_index] = eval_p_derivative(control, t, local_pcof, derivative_order)
                q_control_vals[i,matrix_col_index] = eval_q_derivative(control, t, local_pcof, derivative_order)
            end
        end

        # Convert to MHz/2pi, (or is it 2pi*MHz?)
        if convert_units
            p_control_vals .*= 1e3/(2pi)
            q_control_vals .*= 1e3/(2pi)
        end
        pl = Plots.plot(t_grid, p_control_vals, linecolor=colors_mat, label=labels_mat, lw=linewidth, linestyle=:solid)
        Plots.plot!(t_grid, q_control_vals, linecolor=colors_mat, lw=linewidth, label="", linestyle=:dot)
        Plots.plot!(xlabel="Time (ns)", title="Controls: Derivative $derivative_order")
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
