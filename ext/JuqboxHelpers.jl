module JuqboxHelpers

println("Loading JuqboxHelpers")

using QuantumGateDesign
using LinearAlgebra
using Juqbox


"""
Juqbox Version
"""
function QuantumGateDesign.get_history_convergence(params::Juqbox.objparams, pcof,
        wa::Juqbox.Working_Arrays, N_iterations;
        nsteps_change_factor=2,
        true_history=missing,
        error_limit=1e-15, n_runs=1
    )

    base_nsteps = params.nsteps
    nsteps_change_factor = 2
    order = 2

    histories = []

    # Keeping these as matrices even though they only have one column to be
    # consistent with the main version of this function
    errors_all = Matrix{Float64}(undef, N_iterations, 1)
    timing_all = Matrix{Float64}(undef, N_iterations, 1)
    timing_stddev_all = Matrix{Float64}(undef, N_iterations, 1)
    # Fill with NaN, so that we can break the loop early and still plot, ignoring unfilled values
    errors_all .= NaN
    timing_all .= NaN
    timing_stddev_all .= NaN

    step_sizes = (params.T/base_nsteps) ./ [nsteps_change_factor^k for k in 0:N_iterations-1]

    errors = Vector{Float64}(undef, N_iterations)
    timing = Vector{Float64}(undef, N_iterations)
    timing_stddev = Vector{Float64}(undef, N_iterations)
    errors .= NaN
    timing .= NaN
    timing_stddev .= NaN


    N_derivatives = div(order, 2)

    println("========================================")
    println("Running Order 2 (Juqbox)")
    println("========================================")

    histories_this_order = []

    if ismissing(true_history)
        println("----------------------------------------")
        println("True history not given, using Richardson extrapolation\n")
        println("Calculating solution with base_nsteps=", base_nsteps)
        println("----------------------------------------")
        params.nsteps = base_nsteps

        ret = traceobjgrad(pcof, params, wa, true, false,
                           saveEveryNsteps=div(params.nsteps, base_nsteps))
        base_history = ret[2]

        push!(histories_this_order, base_history)
    end


    for k in 1:N_iterations
        nsteps_multiplier = nsteps_change_factor^k
        params.nsteps = base_nsteps*nsteps_multiplier

        elapsed_times = zeros(n_runs)

        elapsed_times[1] = @elapsed ret = traceobjgrad(
            pcof, params, wa, true, false, saveEveryNsteps=div(params.nsteps, base_nsteps)
        )
        history = ret[2]
        for rerun_i in 2:n_runs
            elapsed_times[rerun_i] = @elapsed ret = traceobjgrad(
                pcof, params, wa, true, false, saveEveryNsteps=div(params.nsteps, base_nsteps)
            )
        end
        mean_elapsed_time = sum(elapsed_times)/length(elapsed_times)
        stddev_elapsed_time = sum((elapsed_times .- mean_elapsed_time) .^ 2)
        stddev_elapsed_time /= length(elapsed_times)-1
        stddev_elapsed_time = sqrt(stddev_elapsed_time)

        push!(histories_this_order, history)

        if ismissing(true_history)
            history_prev = histories_this_order[k]
            error = QuantumGateDesign.richardson_extrap_rel_err(history, history_prev, order)
        else
            error = norm(history - true_history)/norm(true_history)
        end

        errors[k] = error
        timing[k] = mean_elapsed_time
        timing_stddev[k] = stddev_elapsed_time

        println("Nsteps = ", params.nsteps)
        println("Error = ", error)
        println("Mean Elapsed Time = ", mean_elapsed_time)
        println("StdDev Elapsed Time = ", stddev_elapsed_time)
        println("----------------------------------------")

        # Break once we reach high enough precision
        if error < error_limit 
            break
        end
    end

    push!(histories, histories_this_order)

    errors_all[:,1] .= errors
    timing_all[:,1] .= timing
    timing_stddev_all[:,1] .= timing_stddev

    params.nsteps = base_nsteps

    return step_sizes, errors_all, timing_all, timing_stddev_all
    #return step_sizes, errors_all, timing_all, timing_stddev_all, histories
end

end # module JuqboxHelpers
