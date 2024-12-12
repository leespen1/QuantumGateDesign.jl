using Plots 
using JLD2
using QuantumGateDesign

"""
Remove outliers based on a moving average with taken using `Npoints` points on
either side of each datapoint.

Input data should always positive.
"""
function remove_outliers(data, Npoints=10)
    processed_data = similar(data)
    N = length(data)
    for index in 1:N
        avg = 0.0
        n_points_used = 0
        for k in max(1,index-Npoints):min(N,index+Npoints)
            n_points_used += 1
            avg += data[k]
        end
        avg /= n_points_used

        data_val = data[index]
        # Use NaN if data is 10 times the moving average
        processed_data[index] = (avg > 10*data_val) ? NaN : data_val
    end
    return processed_data
end

directories = readdir(".")
# \d+ matches one or more digits
# Other group captures a scientific notation number, or an integer (tries scientific notation first)
# (?:-----) indicates non-capturing group
dir_pattern = r"Order=(\d+)_TargetError=(-?[1-9](?:\.\d+)?[Ee][-+]?\d+|\d+)"

file_pattern = r"""cnot3_opt(?:_juqbox)?_order=(\d+)
    _targetError=(-?[1-9](?:\.\d+)?[Ee][-+]?\d+|\d+)
    _nsteps=(\d+)
    _seed=(\d+)
    _date=(.*) # Capture any character 0 or more times
    \.jld2"""x # 'x' tag ignores whitespace and comments

pls = [plot(ylims=(-8,0)) for i in 1:12]

# For all directories
for dir_entry in directories
    # If the directory name matches the regex pattern
    if occursin(dir_pattern, dir_entry)
        regex_match = match(dir_pattern, dir_entry)
        order = regex_match[1]
        target_error = regex_match[2]
        #println("Directory match! Order=", order, ". TargetError=", target_error, ". Directory name is", dir_entry)

        subdirectories = readdir(dir_entry)
        for subdir_entry in subdirectories
            if occursin(file_pattern, subdir_entry)
                regex_match2 = match(file_pattern, subdir_entry)
                order = parse(Int, regex_match2[1])
                target_error = parse(Float64, regex_match2[2])
                nsteps = parse(Int, regex_match2[3])
                seed = parse(Int, regex_match2[4])
                date = regex_match2[5]

                if (seed == 9) && (target_error == 1e-5)

                    println("File match!\n\tOrder=", order, "\n\tTargetError=", target_error,
                            "\n\tnsteps=", nsteps, "\n\tseed=", seed, "\n\tdate=", date)

                    if (target_error == 0) # Special Juqbox Case
                        infidelities = JLD2.load(dir_entry * "/" * subdir_entry)["primaryHist"]
                        times = JLD2.load(dir_entry * "/" * subdir_entry)["timeHist"]
                        display(diff(times))
                        #display(JLD2.load(dir_entry * "/" * subdir_entry))
                    else
                        opt_history = QuantumGateDesign.read_optimization_history(dir_entry * "/" * subdir_entry)
                        times = opt_history.wall_time ./ 3600 # Time in hours
                        infidelities = opt_history.infidelity
                        println("\t# Iterations=", length(infidelities))
                        #infidelities = opt_history.ipopt_obj_value
                    end


                    #infidelities = replace(x -> x < 0 ? NaN : x, infidelities)
                    infidelities = replace(abs, infidelities)
                    infidelities = remove_outliers(infidelities)
                    log10_infidelities = log10.(infidelities)
                    
                    plot!(pls[order], times, log10_infidelities, title="Order=$order", linewidth=1)
                    #plot!(pls[order], log10_infidelities, title="Order=$order", linewidth=1)
                end
            end
        end
    end
end
#pl = plot(pls[2], pls[4], pls[6], pls[8], pls[10], pls[12], ylabel="Infidelity", xlabel="Wall Time (hrs)", link=:all)
pl = plot(pls[2], pls[4], pls[6], pls[8], pls[10], pls[12], ylabel="Infidelity", xlabel="# Iterations", link=:all)

