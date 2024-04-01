module JuqboxHelpers

println("Loading JuqboxHelpers")

using QuantumGateDesign
using LinearAlgebra
using Juqbox


"""
Juqbox Version
"""
function QuantumGateDesign.get_histories(params::Juqbox.objparams,
        wa::Juqbox.Working_Arrays, pcof, N_iterations;
        min_error_limit=-Inf, max_error_limit=Inf,
        base_nsteps=missing, nsteps_change_factor=2,
    )

    original_nsteps = params.nsteps
    orders = (2,)

    # Default to whatever nsteps the problem original had
    if ismissing(base_nsteps)
        base_nsteps = original_nsteps
    end

    ret_vec = []

    println("Beginning at ", QuantumGateDesign.Dates.now())

    for order in orders
        println("#"^40, "\n")
        println("Juqbox")
        println("Order ", order, "\n")
        println("#"^40)

        nsteps_vec = Int64[]
        step_sizes = Float64[]
        elapsed_times = Float64[]
        histories = Array{Float64, 3}[]
        richardson_errors = Float64[]

        for k in 1:N_iterations
            println("Starting iteration ", k, " at ", QuantumGateDesign.Dates.now())
            nsteps_multiplier = nsteps_change_factor^(k-1)
            params.nsteps = base_nsteps*nsteps_multiplier

            # Run forward evolution
            elapsed_time = @elapsed ret = traceobjgrad(
                pcof, params, wa, true, false,
                saveEveryNsteps=nsteps_multiplier
            )
            history = ret[2]
            # Reorder Juqbox indices to match QuantumGateDesign
            history = permutedims(history, (1,3,2))
            # Convert from complex to real valued
            history = QuantumGateDesign.complex_to_real(history)

            # Compute Richardson Error
            richardson_err = NaN
            if (k > 1)
                history_prev = histories[k-1]
                richardson_err = QuantumGateDesign.richardson_extrap_rel_err(history, history_prev, order)
            end

            push!(nsteps_vec, params.nsteps)
            push!(step_sizes, params.T / params.nsteps)
            push!(elapsed_times, elapsed_time)
            push!(histories, history)
            push!(richardson_errors, richardson_err)

            println("Finished iteration ", k, " at ", QuantumGateDesign.Dates.now())
            println("Nsteps = \t", params.nsteps)
            println("Richardson Error = \t", richardson_err)
            println("Elapsed Time = \t", elapsed_time)
            println("----------------------------------------")

            # Break once we reach high enough precision
            if richardson_err < min_error_limit 
                break
            end

            # If we are reasonably precise, break if the error increases twice
            # (numerical saturation)
            if k > 2
                if ((richardson_err < max_error_limit) 
                    && (error > richardson_errors[k-1]) 
                    && (richardson_errors[k-1] > richardson_errors[k-2])
                   )
                    break
                end
            end
        end
        push!(ret_vec, (order, nsteps_vec, step_sizes, elapsed_times, histories, richardson_errors))
    end
    
    println("Finished at ", QuantumGateDesign.Dates.now())
    println("Returning (order, nsteps_vec, step_sizes, elapsed_times, histories, richardson_errors) for each order.")

    # Make sure to return nsteps to original value
    params.nsteps = original_nsteps

    return ret_vec
end

end # module JuqboxHelpers
