# I should include a manufactured solution in this

"""
Double the step size to get more precise solutions, and compare the change in
error with the "true" solution (solution using the most steps / smallest step
size).

Error is Frobenius norm of the difference between the "true" and approximate
solutions, with the number of points in time compared taken to be the number
of points in time when using the fewest steps / largest step size.
"""
function plot_history_convergence(prob, control, pcof, N_iterations;
        orders=(2, 4), nsteps_change_factor=2,
        return_data=false, duration_error=true
    )
    base_nsteps = prob.nsteps

    # Copy problem so we can mutate nsteps without altering input
    prob_copy = copy(prob)
    histories = Vector[]

    pl = Plots.plot(xlabel="Step Size Δt", ylabel="Relative Error in State Vector History", scale=:log10)
    yticks = [10.0 ^ n for n in -15:15] 
    Plots.plot!(pl, yticks=yticks, legend=:topleft)

    # Get "true" solution using many timesteps with highest order method
    most_steps = base_nsteps*(nsteps_change_factor ^ N_iterations)
    prob_copy.nsteps = most_steps
    true_history = eval_forward(prob_copy, control, pcof, order=orders[end])

    # Parse history to include only times included when using base_nsteps
    true_history = true_history[:,1:(nsteps_change_factor^N_iterations):end,:]

    errors_all = Matrix{Float64}(undef, N_iterations, 2*length(orders))

    step_sizes = (prob.tf/base_nsteps) ./ [nsteps_change_factor^k for k in 0:N_iterations-1]

    try
    for (j, order) in enumerate(orders)
        println("Order $j")
        errors = Vector{Float64}(undef, N_iterations)

        for k in 1:N_iterations
            nsteps_multiplier = nsteps_change_factor^(k-1)
            prob_copy.nsteps = base_nsteps*nsteps_multiplier
            history = eval_forward(prob_copy, control, pcof, order=order)
            # Skip over steps to match base_nsteps solution
            history = history[:,1:nsteps_multiplier:end,:]

            if duration_error # Use error across entire duration of problem
                error = norm(history - true_history)/norm(true_history)
            else # Only use error at final state
                error = norm(history[:,end,:] - true_history[:,end,:])/norm(true_history[:,end,:])
            end

            println("Iteration $k error: ", error)
            errors[k] = error
        end

        Plots.plot!(pl, step_sizes, errors, marker=:circle, markersize=5, label="Order=$order")

        errors_all[:,j] .= errors
        #errors_all[:,length(orders)+j] .= order_line
    end
    # If interrupted by keyboard, stop early but still finish the graph
    catch e
        if e isa InterruptException
            println("Keyboard interruption, ending early")
        # If not keyboard interruption, still throw the error
        else
            throw(e)
        end
    end

    order_line2 = step_sizes .^ 2
    order_line2 .*= 2 * errors_all[1,1]/order_line2[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line2, label="Δt²", linecolor=:black, linestyle=:dash)

    order_line4 = step_sizes .^ 4
    order_line4 .*= 2 * errors_all[1,2]/order_line4[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line4, label="Δt⁴", linecolor=:black, linestyle=:dashdot)


    if return_data
        return step_sizes, errors_all
    end
    return pl
end

function plot_history_convergence_new(prob, control, pcof, N_iterations;
        orders=(2, 4), nsteps_change_factor=2,
        return_data=false, duration_error=true
    )
    base_nsteps = prob.nsteps

    # Copy problem so we can mutate nsteps without altering input
    prob_copy = copy(prob)
    histories = Vector[]

    pl = Plots.plot(xlabel="Step Size Δt", ylabel="Relative Error in State Vector History", scale=:log10)
    yticks = [10.0 ^ n for n in -15:15] 
    Plots.plot!(pl, yticks=yticks, legend=:topleft)

    # Get "true" solution using many timesteps with highest order method
    most_steps = base_nsteps*(nsteps_change_factor ^ N_iterations)
    prob_copy.nsteps = most_steps
    true_history = HermiteOptimalControl.eval_forward_arbitrary_order(
        prob_copy, control, pcof, order=orders[end]
    )

    # Parse history to include only times included when using base_nsteps
    true_history = true_history[:,1,1:(nsteps_change_factor^N_iterations):end,:]

    errors_all = Matrix{Float64}(undef, N_iterations, 2*length(orders))

    step_sizes = (prob.tf/base_nsteps) ./ [nsteps_change_factor^k for k in 0:N_iterations-1]

    try
    for (j, order) in enumerate(orders)
        errors = Vector{Float64}(undef, N_iterations)

        for k in 1:N_iterations
            nsteps_multiplier = nsteps_change_factor^(k-1)
            prob_copy.nsteps = base_nsteps*nsteps_multiplier
            history = HermiteOptimalControl.eval_forward_arbitrary_order(prob_copy, control, pcof, order=order)
            # Skip over steps to match base_nsteps solution
            history = history[:,1,1:nsteps_multiplier:end,:]

            if duration_error # Use error across entire duration of problem
                error = norm(history - true_history)/norm(true_history)
            else # Only use error at final state
                error = norm(history[:,end,:] - true_history[:,end,:])/norm(true_history[:,end,:])
            end

            errors[k] = error
        end

        Plots.plot!(pl, step_sizes, errors, marker=:circle, markersize=5, label="Order=$order")

        errors_all[:,j] .= errors
        #errors_all[:,length(orders)+j] .= order_line
    end
    # If interrupted by keyboard, stop early but still finish the graph
    catch e
        if e isa InterruptException
            println("Keyboard interruption, ending early")
        # If not keyboard interruption, still throw the error
        else
            throw(e)
        end
    end

    # Order lines may extend too far down. Save the old limits so I can fix the window size back.
    old_ylims = ylims(pl)

    order_line2 = step_sizes .^ 2
    order_line2 .*= 2 * errors_all[1,1]/order_line2[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line2, label="Δt²", linecolor=:black, linestyle=:dash)

    order_line4 = step_sizes .^ 4
    order_line4 .*= 2 * errors_all[1,2]/order_line4[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line4, label="Δt⁴", linecolor=:black, linestyle=:dashdot)

    order_line6 = step_sizes .^ 6
    order_line6 .*= 2 * errors_all[1,3]/order_line6[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line6, label="Δt⁶", linecolor=:black, linestyle=:dot)

    order_line8 = step_sizes .^ 8
    order_line8 .*= 2 * errors_all[1,4]/order_line8[1] # Adjust vertical position to match data, with small offset for visibility
    Plots.plot!(pl, step_sizes, order_line8, label="Δt⁸", linecolor=:black, linestyle=:solid)

    ylims!(pl, old_ylims...)

    if return_data
        return step_sizes, errors_all
    end
    return pl
end
#=
detuning_frequency    = 1.0
self_kerr_coefficient = 0.5
N_ess_levels = 2
N_guard_levels = 0
tf = 1.0
nsteps = 5
prob = HermiteOptimalControl.single_transmon_qubit_rwa(
    detuning_frequency, self_kerr_coefficient, N_ess_levels,
    N_guard_levels, tf, nsteps
)

N_coeff_per_control = 4
control = HermiteOptimalControl.sincos_control(4)

pcof = ones(8)

N_iterations = 13

pl = HermiteOptimalControl.plot_history_convergence(prob, control, pcof, N_iterations)
=#
