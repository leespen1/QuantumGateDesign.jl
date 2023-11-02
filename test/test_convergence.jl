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
        orders=[2, 4], nsteps_change_factor=2
    )
    base_nsteps = prob.nsteps

    # Copy problem so we can mutate nsteps without altering input
    prob_copy = copy(prob)
    histories = Vector[]

    pl = Plots.plot(xlabel="Step Size Δt", ylabel="Error", scale=:log10)
    yticks = [10.0 ^ n for n in -15:15] 
    Plots.plot!(pl, yticks=yticks, legend=:topleft)

    # Get "true" solution using many timesteps with highest order method
    most_steps = base_nsteps*(nsteps_change_factor ^ N_iterations)
    prob_copy.nsteps = most_steps
    true_history = eval_forward(prob_copy, control, pcof, order=orders[end])

    # Parse history to include only times included when using base_nsteps
    true_history = true_history[:,1:(nsteps_change_factor^N_iterations):end,:]

    for order in orders
        errors = Vector{Float64}(undef, N_iterations)

        for k in 1:N_iterations
            nsteps_multiplier = nsteps_change_factor^(k-1)
            prob_copy.nsteps = base_nsteps*nsteps_multiplier
            history = eval_forward(prob_copy, control, pcof, order=order)
            # Skip over steps to match base_nsteps solution
            history = history[:,1:nsteps_multiplier:end,:]

            error = norm(history - true_history)
            errors[k] = error

        end

        step_sizes = prob.tf ./ [nsteps_change_factor^k for k in 0:N_iterations-1]

        Plots.plot!(pl, step_sizes, errors, marker=:circle, markersize=5, label="Order=$order")
        Plots.plot!(pl, step_sizes, step_sizes .^ order, label="Δt^$order")
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
