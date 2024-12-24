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

"""
Function to extract iteration numbers where alpha_pr ends with 'w' (for watchdog)
"""
function extract_watchdog_iterations(log_filename::String)
    iteration_numbers = Int[]

    input_string = read(log_filename, String)
    lines = split(input_string, "\n")

    # Find the start of the ipopt section of looking for header
    header = "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls"
    start_line = findfirst(line -> line == header, lines)

    for line in lines[start_line:end] # TODO change end to be at the closing statement of IPOPT or something
        # Only do iteration lines (which start with whitespace, followed by digits)
        if occursin(r"^\s*\d+", line)
            columns = split(line)

            if length(columns) == 10
                alpha_pr = columns[end-1]

                # Only push iteration number if it is not for a watchog phase (alpha_pr doesn't end with 'w')
                if !endswith(alpha_pr, "w")
                    # Extract the iteration number from the beginning of the line
                    iteration = parse(Int, columns[1])
                    push!(iteration_numbers, iteration)
                end
            end
        end
    end

    return iteration_numbers
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
        order_str = regex_match[1]
        target_error_str = regex_match[2]
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

                    #TODO Add processing to determing if in watchdog phase
                    log_pattern = Regex("cnot3_optimization_order=$(order_str)_targeterror=$(target_error_str)_seed=$(seed).*\\.log")
                    log_files = filter(f -> occursin(log_pattern, f), subdirectories)
                    display(log_pattern)
                    display(log_files)
                    @assert length(log_files) == 1
                    log_file = first(log_files) # There should 

                    iteration_numbers = extract_watchdog_iterations(dir_entry * "/" * log_file)
                    iteration_numbers .+= 1

                    
                    iter_count = opt_history.iter_count

                    #infidelities = replace(x -> x < 0 ? NaN : x, infidelities)
                    infidelities = replace(abs, infidelities)
                    log10_infidelities = log10.(infidelities)
                    
                    plot!(pls[order], iter_count, log10_infidelities, title="Order=$order", linewidth=1)
                    #plot!(pls[order], times[iteration_numbers], log10_infidelities[iteration_numbers], title="Order=$order", linewidth=1)
                    #plot!(pls[order], log10_infidelities, title="Order=$order", linewidth=1)
                end
            end
        end
    end
end
#pl = plot(pls[2], pls[4], pls[6], pls[8], pls[10], pls[12], ylabel="Infidelity", xlabel="Wall Time (hrs)", link=:all)
pl = plot(pls[2], pls[4], pls[6], pls[8], pls[10], pls[12], ylabel="Infidelity", xlabel="# Iterations", link=:all)

