# I should include a manufactured solution in this

"""
Double the step size to get more precise solutions, and compare the change in
error with the "true" solution (solution using the most steps / smallest step
size).

Error is Frobenius norm of the difference between the "true" and approximate
solutions, with the number of points in time compared taken to be the number
of points in time when using the fewest steps / largest step size.

Note: Plots.scalefontsizes(1.5-2.0) seems appropriate for slideshow
"""
function plot_history_convergence(prob, control, pcof, N_iterations;
        orders=(2, 4, 6, 8), nsteps_change_factor=2,
        duration_error=true,
        include_orderlines=true, fontsize=16, true_history=missing,
        error_limit=1e-15, n_runs=1
    )
    base_nsteps = prob.nsteps

    # Copy problem so we can mutate nsteps without altering input
    prob_copy = copy(prob)
    histories = Vector[]

    pl_stepsize = Plots.plot(ylabel="Log₁₀(Rel Err)", fontsize=fontsize)
    pl_timing = Plots.plot(ylabel="Log₁₀(Rel Err)", fontsize=fontsize)
    Plots.plot!(pl_stepsize, xlabel="Log₁₀(Step Size Δt)")
    Plots.plot!(pl_timing, xlabel="Log₁₀(Elapsed Time) (s)")

    #yticks = [10.0 ^ n for n in -15:15] # Log10 scale version
    yticks = -15:15 
    Plots.plot!(pl_stepsize, yticks=yticks, legend=:outerright)
    Plots.plot!(pl_timing, yticks=yticks, legend=:outerright)

    # Get "true" solution using many timesteps with highest order method
    if ismissing(true_history)
        println("True history not provided, using 'fine-grain' numerical solution")
        most_steps = base_nsteps*(nsteps_change_factor ^ N_iterations)
        prob_copy.nsteps = most_steps
        true_history = eval_forward(
            prob_copy, control, pcof, order=orders[end]
        )

        # Parse history to include only times included when using base_nsteps
        true_history = true_history[:,1,1:(nsteps_change_factor^N_iterations):end,:]
    end


    errors_all = Matrix{Float64}(undef, N_iterations, length(orders))
    timing_all = Matrix{Float64}(undef, N_iterations, length(orders))
    timing_stddev_all = Matrix{Float64}(undef, N_iterations, length(orders))
    # Fill with NaN, so that we can break the loop early and still plot, ignoring unfilled values
    errors_all .= NaN
    timing_all .= NaN
    timing_stddev_all .= NaN

    step_sizes = (prob.tf/base_nsteps) ./ [nsteps_change_factor^k for k in 0:N_iterations-1]


    N_initial_conditions = size(prob.u0, 2)

    try
    for (j, order) in enumerate(orders)
        errors = Vector{Float64}(undef, N_iterations)
        timing = Vector{Float64}(undef, N_iterations)
        timing_stddev = Vector{Float64}(undef, N_iterations)
        errors .= NaN
        timing .= NaN
        timing_stddev .= NaN


        N_derivatives = div(order, 2)

        println("========================================")
        println("Running Order ", order)
        println("========================================")
        for k in 1:N_iterations
            nsteps_multiplier = nsteps_change_factor^(k-1)
            prob_copy.nsteps = base_nsteps*nsteps_multiplier

            elapsed_times = zeros(n_runs)
            for rerun_i in 1:n_runs
                elapsed_times[rerun_i] = @elapsed history = eval_forward(prob_copy, control, pcof, order=order)
            end
            mean_elapsed_time = sum(elapsed_times)/length(elapsed_times)
            stddev_elapsed_time = sum((elapsed_times .- mean_elapsed_time) .^ 2)
            stddev_elapsed_time /= length(elapsed_times)-1
            stddev_elapsed_time = sqrt(stddev_elapsed_time)


            # Skip over steps to match base_nsteps solution
            history = history[:,1,1:nsteps_multiplier:end,:]

            if duration_error # Use error across entire duration of problem
                error = norm(history - true_history)/norm(true_history)
            else # Only use error at final state
                error = norm(history[:,end,:] - true_history[:,end,:])/norm(true_history[:,end,:])
            end

            errors[k] = error
            timing[k] = mean_elapsed_time
            timing_stddev[k] = stddev_elapsed_time

            println("Nsteps = ", prob_copy.nsteps)
            println("Error = ", error)
            println("Mean Elapsed Time = ", mean_elapsed_time)
            println("StdDev Elapsed Time = ", stddev_elapsed_time)
            println("----------------------------------------")

            # Break once we reach high enough precision
            if error < error_limit 
                break
            end
        end


        Plots.scatter!(pl_timing, log10.(timing), log10.(errors), marker=:circle, markersize=5, label="Order=$order")
        Plots.plot!(pl_stepsize, log10.(step_sizes), log10.(errors), marker=:circle, markersize=5, label="Order=$order")

        errors_all[:,j] .= errors
        timing_all[:,j] .= timing
        timing_stddev_all[:,j] .= timing_stddev
        #errors_all[:,length(orders)+j] .= order_line
    end

    stepsize_xlims = collect(Plots.xlims(pl_stepsize))
    timing_xlims   = collect(Plots.xlims(pl_timing))
    Plots.plot!(pl_stepsize, stepsize_xlims, [-7, -7], linecolor=:red, label="Target Error")
    # If interrupted by keyboard, stop early but still finish the graph
    catch e
        if e isa InterruptException
            println("Keyboard interruption, ending early")
        # If not keyboard interruption, still throw the error
        else
            throw(e)
        end
    end


    # Add order lines
    if include_orderlines
        # Order lines may extend too far down. Save the old limits so I can fix the window size back.
        old_ylims = Plots.ylims(pl_stepsize)

        linestyle_list = (:solid, :dash, :dot, :dashdot)
        for (j, order) in enumerate(orders)
            order_line = step_sizes .^ order
            order_line .*= 2 * errors_all[1,j]/order_line[1] # Adjust vertical position to match data, with small offset for visibility

            linestyle_index = j % length(linestyle_list)
            if linestyle_index > 0
                linestyle = linestyle_list[linestyle_index]
            else 
                linestyle = linestyle_list[end]
            end

            Plots.plot!(pl_stepsize, log10.(step_sizes), log10.(order_line), label="Δt^$order", linecolor=:black, linestyle=linestyle)
        end

        Plots.ylims!(pl_stepsize, old_ylims...)
        Plots.plot!(pl_stepsize, legend=:outerright)
    end

    return pl_stepsize, pl_timing, step_sizes, errors_all, timing_all, timing_stddev_all
end

function just_plot_history_convergence(step_sizes, errors_all, timing_all, timing_stddev_all;
        orders=(2, 4, 6, 8, 10), include_orderlines=true, fontsize=16, orderline_offset=0
    )

    pl_stepsize = Plots.plot(ylabel="Log₁₀(Rel Err)", fontsize=fontsize)
    pl_timing = Plots.plot(ylabel="Log₁₀(Rel Err)", fontsize=fontsize)
    Plots.plot!(pl_stepsize, xlabel="Log₁₀(Step Size Δt)")
    Plots.plot!(pl_timing, xlabel="Log₁₀(Elapsed Time) (s)")

    yticks = -15:15 
    Plots.plot!(pl_stepsize, yticks=yticks, legend=:outerright)
    Plots.plot!(pl_timing, yticks=yticks, legend=:outerright)

    labels = ["Order=$order" for order in orders]
    labels = reshape(labels, 1, :)
    Plots.plot!(pl_timing, log10.(timing_all), log10.(errors_all), marker=:circle, markersize=5, labels=labels, linealpha=0.5)
    Plots.plot!(pl_stepsize, log10.(step_sizes), log10.(errors_all), marker=:circle, markersize=5, labels=labels)

    stepsize_xlims = collect(Plots.xlims(pl_stepsize))
    timing_xlims   = collect(Plots.xlims(pl_timing))
    Plots.plot!(pl_stepsize, stepsize_xlims, [-7, -7], linecolor=:red, label="Target Error")

    # Add order lines
    if include_orderlines
        # If a single number is given, fill vector with that number
        if length(orderline_offset) == 1
            orderline_offset = ones(length(orders)) .* orderline_offset
        end
        @assert length(orderline_offset) == length(orders)

        # Order lines may extend too far down. Save the old limits so I can fix the window size back.
        old_ylims = Plots.ylims(pl_stepsize)

        linestyle_list = (:solid, :dash, :dot, :dashdot)
        for (j, order) in enumerate(orders)
            order_line = step_sizes .^ order
            order_line .*= 2 * errors_all[1,j]/order_line[1] # Adjust vertical position to match data, with small offset for visibility

            linestyle_index = j % length(linestyle_list)
            if linestyle_index > 0
                linestyle = linestyle_list[linestyle_index]
            else 
                linestyle = linestyle_list[end]
            end

            Plots.plot!(
                pl_stepsize, log10.(step_sizes), log10.(order_line) .+ orderline_offset[j],
                label="Δt^$order", linecolor=:black, linestyle=linestyle
            )
        end

        Plots.ylims!(pl_stepsize, old_ylims...)
        Plots.plot!(pl_stepsize, legend=:outerright)
    end
    return pl_stepsize, pl_timing
end
