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

"""
    plot_populations(history; [ts=missing, level_indices=missing, labels=missing])

Given state vector history, plot population evolution (assuming single qubit for labels).
"""
function plot_populations(history::AbstractArray{Float64, 4}; ts=missing,
        level_indices=missing, labels=missing)
    if ismissing(ts)
        ts = 0:size(history, 3)-1
        xlabel = "Timestep #"
    else
        xlabel = "Time (ns)"
    end

    if ismissing(level_indices)
        complex_system_size = div(size(history, 1), 2)
        level_indices = 1:complex_system_size
    end

    if ismissing(labels)
        labels = ["Level $i" for i in level_indices]
    end
    labels = reshape(labels, 1, :) # Labels must be a row matrix

    
    populations = get_populations(history)
    N_levels = size(populations, 1)


    ret = []
    # Iterate over initial conditions
    for initial_condition in 1:size(populations, 3)
        title = "Initial Condition $initial_condition"
        pl = Plots.plot(xlabel=xlabel, ylabel="Population", 
                        title=title, legend=:outerright)
        # Iterate over essential states
        for (i, level_index) in enumerate(level_indices)
            Plots.plot!(pl, ts, populations[level_index, :, initial_condition],
                  label=labels[i])
        end
        push!(ret, pl)
    end
    return ret
end

"""
Like plot_populations, but plotting the real and imaginary parts of the state
vector, instead of the population
"""
function plot_states(history::AbstractArray{Float64, 4}; ts=missing,
        level_indices=missing, labels=missing)
    if ismissing(ts)
        ts = 0:size(history, 3)-1
        xlabel = "Timestep #"
    else
        xlabel = "Time (ns)"
    end

    if ismissing(level_indices)
        complex_system_size = div(size(history, 1), 2)
        level_indices = 1:complex_system_size
    end

    if ismissing(labels)
        re_labels = ["Level $i (real)" for i in level_indices]
        im_labels = ["Level $i (imag)" for i in level_indices]
    end


    ret = []
    # Iterate over initial conditions
    for initial_condition in 1:size(history, 4)
        title = "Initial Condition $initial_condition"
        pl = Plots.plot(xlabel=xlabel, ylabel="Population", 
                        title=title, legend=:outerright)
        # Iterate over essential states
        for (i, level_index) in enumerate(level_indices)
            Plots.plot!(pl, ts, history[level_index, 1, :, initial_condition],
                  label=re_labels[i], linecolor=i)
            Plots.plot!(pl, ts, history[level_index+complex_system_size, 1, :, initial_condition],
                  label=im_labels[i], linecolor=i, linestyle=:dash)
        end
        push!(ret, pl)
    end
    return ret
end


"""
Do the "gradient agreement test" and plot results
"""
function plot_gradient_agreement(prob, controls, target; 
        orders=(2,4,6,8,10), cost_type=:Infidelity,
        n_runs=10, amax=5e-2, abstol=1e-15, reltol=1e-15
    )

    N_orders = length(orders)
    N_coeffs = get_number_of_control_parameters(controls)


    gradients = Array{Float64, 4}(undef, N_coeffs, 3, N_orders, n_runs)
    errors = Array{Float64, 3}(undef, n_runs, N_orders, 2)


    for i in 1:n_runs
        pcof = rand(MersenneTwister(i), N_coeffs) .* amax
        for (k, order) in enumerate(orders)
            # Check that gradients calculated using discrete adjoint and finite difference
            # methods agree to reasonable precision
            grad_disc_adj = discrete_adjoint(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            grad_forced = eval_grad_forced(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            grad_fin_diff = eval_grad_finite_difference(
                prob, controls, pcof, target, order=order,
                cost_type=cost_type, abstol=abstol=abstol, reltol=reltol
            )

            gradients[:,1,k,i] .= grad_disc_adj
            gradients[:,2,k,i] .= grad_forced
            gradients[:,3,k,i] .= grad_fin_diff

            errors[i, k, 1] = norm(grad_forced - grad_disc_adj)/norm(grad_disc_adj)
            errors[i, k, 2] = norm(grad_fin_diff - grad_disc_adj)/norm(grad_disc_adj)
        end
    end

    replace_zero_with_epsilon(x) = (x == 0.0) ? 1e-16 : x
    errors = replace_zero_with_epsilon.(errors)
    errors = log10.(errors)

    xticks = 1:n_runs
    yticks = -20:20

    display(errors)

    pl = Plots.plot(xlabel="Random Control Vector #", ylabel="Log₁₀(Rel Err in Gradient)",
                    legend=:outerright, xticks=xticks, yticks=yticks, size=(600,400))

    colors = Plots.theme_palette(:auto)
    marker_forced = :circle
    marker_fin_diff = :star5
    for (k, order) in enumerate(orders)
        Plots.scatter!(pl, errors[:, k, 1], color=colors[k], label="Order $order, Forced", marker=marker_forced, markersize=5)
        Plots.scatter!(pl, errors[:, k, 2], color=colors[k], label="Order $order, Fin Diff", marker=marker_fin_diff, markersize=5)
    end
    Plots.plot!(pl, yticks=-16:0)


    return pl, errors
end
