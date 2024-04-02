module JuqboxHelpers

println("Loading JuqboxHelpers")

using QuantumGateDesign
using LinearAlgebra
using OrderedCollections
using Dates
using Juqbox
using JLD2


"""
Juqbox Version
"""
function QuantumGateDesign.get_histories(params::Juqbox.objparams,
        wa::Juqbox.Working_Arrays, pcof, N_iterations;
        min_error_limit=-Inf, max_error_limit=Inf, base_nsteps=missing, 
        nsteps_change_factor=2, start_iteration=1, jld2_filename=missing
    )

    # Touch file, create if needed (append mode, so existing file isn't wiped)
    if !ismissing(jld2_filename)
        JLD2.jldopen(jld2_filename, "a") do f end
    end

    original_nsteps = params.nsteps
    orders = (2,)

    # Default to whatever nsteps the problem original had
    if ismissing(base_nsteps)
        base_nsteps = original_nsteps
    end

    ret_dict = OrderedCollections.OrderedDict()

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

        summary_dict = Dict(
            "order" => order,
            "nsteps" => nsteps_vec,
            "step_sizes" => step_sizes,
            "elapsed_times" => elapsed_times,
            "histories" => histories,
            "richardson_errors" => richardson_errors
        )

        summary_dict_name = "Order $order (Juqbox)"
        ret_dict[summary_dict_name] = summary_dict

        if !ismissing(jld2_filename)
            JLD2.jldopen(jld2_filename, "a") do f end
            jld2_dict = JLD2.load(jld2_filename)
            jld2_dict[summary_dict_name] = summary_dict
            JLD2.save(jld2_filename, jld2_dict)
        end

        for k in start_iteration:N_iterations
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

            # Save intermediate results
            if !ismissing(jld2_filename)
                JLD2.jldopen(jld2_filename, "a") do f end
                jld2_dict = JLD2.load(jld2_filename)
                jld2_dict[summary_dict_name] = summary_dict
                JLD2.save(jld2_filename, jld2_dict)
            end

            # Break once we reach high enough precision
            if richardson_errors[end] < min_error_limit 
                println("Breaking early due to precision reached")
                break
            end

            # If we are reasonably precise (error under max_error_limit), break
            # if the error increases twice (numerical saturation)
            if length(richardson_errors) > 2
                if ((richardson_errors[end] < max_error_limit) 
                    && (richardson_errors[end] > richardson_errors[end-1] > richardson_errors[end-2]))
                    println("Breaking early due to numerical saturation")
                    break
                end
            end
        end
    end
    
    println("Finished at ", Dates.now())
    println("Returning Results")

    # Make sure to return nsteps to original value
    params.nsteps = original_nsteps

    return ret_dict
end

end # module JuqboxHelpers
