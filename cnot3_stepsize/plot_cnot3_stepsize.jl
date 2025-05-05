using CairoMakie, LaTeXStrings, Statistics, IterTools, Printf, Format
using DataFrames, DataFramesMeta, CSV, DelimitedFiles
using LinearAlgebra
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

"""
Given a string, try to interpret it as an Int, Float64, Bool, and if none of
those work, default to string.
"""
function val_parse(value::AbstractString)
    pvalue = tryparse(Int, value) # parsed value
    pvalue = isnothing(pvalue) ? tryparse(Float64, value) : pvalue
    pvalue = isnothing(pvalue) ? tryparse(Bool, value) : pvalue
    pvalue = isnothing(pvalue) ? string(value) : pvalue 
    return pvalue
end

function parse_filename_params(filename::String)
    # Extract key=value pairs
    reduced_filename = first(splitext(basename(filename))) # Remove directory and extension
    key_val_regex = r"([a-zA-Z0-9]+)=([^\_]+)"
    #key_val_regex = r"(\w+)=([^\._]+)" # \w+ also include underscores, hence why I don't use
    matches = eachmatch(key_val_regex, reduced_filename)
    return Dict(Symbol(m.captures[1]) => val_parse(m.captures[2]) for m in matches)
end

function load_dataframe_with_metadata(filename::String)
    df = CSV.read(filename, DataFrame; header = true, stripwhitespace = true)
    params = parse_filename_params(filename)
    for (key, value) in params
        df[!, key] .= value
    end
    return df
end

function load_fstates_with_metadata(filepath::String)
    final_states_str = readdlm(filepath, ',', String)
    final_states = map(x -> strip(x) == "NaN + NaN*im" ? NaN + NaN*im : parse(ComplexF64, strip(x)),
                       final_states_str) 

    true_final_state = final_states[end,:]
    true_final_state_size = norm(true_final_state)
    rel_err(x) = norm(x - true_final_state) / true_final_state_size

    n_runs = size(final_states, 1)

    nsteps_vec = [2^i for i in 1:n_runs]
    # "True" final state has no comparison point, so use NaN as error
    relerr_vec = vcat([rel_err(final_states[i,:]) for i in 1:n_runs-1], NaN)

    df = DataFrame(:nsteps => nsteps_vec, :relerr => relerr_vec) 

    params = parse_filename_params(filepath)
    for (key, value) in params
        df[!, key] .= value
    end
    # Return a DataFrame with params and the matrix
    return df
end

"""
Given points (x1,y1) and (x2,y2), find the value of x for which the line going
through the two points goes through y.

Note that I will want to pass the logarithm of the points, so that the linear
interpolation is valid.
"""
function find_x(y, x1, y1, x2, y2)
    m = (y2-y1) / (x2-x1)
    x = x1 + (y-y1)/m
    return x
end

function find_y(x, x1, y1, x2, y2)
    m = (y2-y1) / (x2-x1)
    y = y1 + (x-x1)*m
    return y
end

function table(mat::Matrix{String})
    for row in eachrow(mat)
        println(reduce((x,y) -> x * " & " * y, row), " \\\\")
    end
end

function make_plot(processed_df::DataFrame)
    colors = vcat(Makie.wong_colors()[1:6], :darkkhaki)

    ### Set up Makie Figures, Axes
    ticks_10f(i) = L"10^{%$i}"
    log10_ticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))

    unlabeled_timestep_ticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
    labeled_timestep_ticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
    minor_timestep_ticks = 2 .^ (0:20)
    inch = 96 # Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
    fig = CairoMakie.Figure(size=(6.25inch, 4.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))

    nsteps_vs_error_grid = fig[1,1]
    nsteps_vs_time_grid = fig[2,1]
    error_vs_time_grid = fig[1:2,2]

    ax_nsteps_vs_error = CairoMakie.Axis(
        nsteps_vs_error_grid,
        xscale=CairoMakie.log2,
        yscale=CairoMakie.log10,
        #xlabel="Number of Timesteps",
        ylabel="Mean Final State Error",
        xticks=unlabeled_timestep_ticks,
        xminorticks = minor_timestep_ticks,
        xminorticksvisible = true,
        xminorgridvisible = true,
        yticks=log10_ticks,
        #limits=((2^5, 2^20), (10.0^(-10.25), 10.0^0.25)),
        limits=((2^5, 2^20), (1e-10, 1e0)),
        #limits=((2^5, 2^20), (1e-10, 1e1)),
    )

    ax_nsteps_vs_time = CairoMakie.Axis(
        nsteps_vs_time_grid,
        xscale=CairoMakie.log2,
        yscale=CairoMakie.log10,
        xlabel="Number of Timesteps",
        ylabel="Mean Time to Compute Gradient (s)",
        xticks=labeled_timestep_ticks,
        xminorticks = minor_timestep_ticks,
        xminorticksvisible = true,
        xminorgridvisible = true,
        #xticks=(my_xticks, my_xticklabels),
        yticks=log10_ticks,
        #limits=((2^5, 2^20), (10.0^(-2.25), 10.0^3.25)),
        limits=((2^5, 2^20), (1e-2, 1e3)),
        #title="Elapsed Time Plot",
    )

    # Share the x-axis
    linkxaxes!(ax_nsteps_vs_time, ax_nsteps_vs_error)

    ax_error_vs_time = CairoMakie.Axis(
        error_vs_time_grid,
        xscale=CairoMakie.log10,
        yscale=CairoMakie.log10,
        xlabel="Mean Final State Error",
        ylabel="Mean Time to Compute Gradient (s)",
        #xminorticksvisible = true,
        #xminorgridvisible = true,
        xticks=log10_ticks,
        yticks=log10_ticks,
        yminorticks=IntervalsBetween(10),
        yminorticksvisible=true,
        yminorgridvisible=true,
        #title="Elapsed Time Plot",
        #limits=((10.0^(-10.25), 10.0^(-0.75)), (10.0^(-0.25),10.0^(5.25))),
        #limits=((1e-10, 1e-1), (10.0^(-0.25),10.0^(5.25))),
        limits=((1e-10, 1e-1), (1e0,1e5)),
        #limits=((1e-10, 1e0), (1e0,1e5)),
    )

    grouped_df = @groupby(processed_df, :usejuqbox, :order)
    for (group_i, subdf) in enumerate(grouped_df)
        order = subdf[1,:order]
        usejuqbox = subdf[1,:usejuqbox]

        if usejuqbox
            if order != 2
                throw("Data processing error. usejuqbox=true, but Order≠2.")
            end
            label = "Order 2 (St\u00f6rmer-Verlet)"
        else
            label = "Order $order"
        end

        ##### Make Plots
        ### Number of Timesteps vs Elapsed Time
        lines_obj = lines!(
            ax_nsteps_vs_time, subdf[:, :nsteps], subdf[:, :elapsed_time_mean];
            color=colors[group_i], label=label
        )
        CairoMakie.translate!(lines_obj, 0, 0, -group_i) # Put the lines in the right z-order

        ### Number of Timesteps vs Relative Error in Final State
        lines_obj = lines!(
            ax_nsteps_vs_error, subdf[:, :nsteps], subdf[:, :relerr_mean];
            color=colors[group_i], label=label
        )
        CairoMakie.translate!(lines_obj, 0, 0, -group_i) # Put the lines in the right z-order

        ### Relative Error vs Elapsed Time
        # cut off the non-asymptotic results

        
        cutoff_limit = 1e0
        cutoff_index = findlast(x -> x > cutoff_limit, subdf[:,:relerr_mean]) 
        while isnothing(cutoff_index)
            cutoff_limit /= 10
            cutoff_index = findlast(x -> x > cutoff_limit, subdf[:,:relerr_mean]) 
        end

        lines!(
            ax_error_vs_time,
            subdf[cutoff_index:end, :relerr_mean],
            subdf[cutoff_index:end, :elapsed_time_mean];
            color=colors[group_i], label=label
        )

    end

    # Add a hard-coded line to approximately extrapolate 2nd order Hemite and Stormer-Verlet
    #dummy_relerr_mean = [10.0 ^(-i) for i in 4.25:11]
    #dummy_elapsedtime_mean = 1.1 .* (dummy_relerr_mean .^ -0.47)
    dummy_relerr_mean = [10.0 ^(-i) for i in 3.5:11]
    dummy_elapsedtime_mean = 3.0 .* (dummy_relerr_mean .^ -0.47)
    #dummy_relerr_mean = [10.0 ^(-i) for i in 1.25:11]
    #dummy_elapsedtime_mean = 35 .* (dummy_relerr_mean .^ -0.47)
    lines!(
        ax_error_vs_time, dummy_relerr_mean, dummy_elapsedtime_mean;
        color=:black, linestyle=:dot, label="Order 2 (Approximate Extrapolation)"
    )
        
    # Add legend to very end
    Legend(fig[end+1,:], ax_error_vs_time, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
    #rowsize!(fig.layout, 1, Relative(0.6))
    rowgap!(fig.layout, 1, 0.125inch) # Minimize space between legend and plot area.
    rowgap!(fig.layout, 2, 0.05inch) # Minimize space between legend and plot area.
    colgap!(fig.layout, 1, 0.15inch) # Minimize space between legend and plot area.

    return fig
end

function make_speedup_table(processed_df::DataFrame)
    target_errors = collect(-1:-1:-7)

    grouped_df = @groupby(processed_df, :usejuqbox, :order)
    n_groups = length(grouped_df)
    target_log2nsteps_mat = fill(NaN, length(target_errors), n_groups)
    target_log10time_mat = fill(NaN, length(target_errors), n_groups)

    for (group_i, subdf) in enumerate(grouped_df)
        # Convert to log scale, cut off end (where relerr is NaN, no reference point)
        log2_nsteps_vec = log2.(subdf[1:end-1,:nsteps])
        log10_relerr_vec = log10.(subdf[1:end-1,:relerr_mean])
        log10_elapsedtime_vec = log10.(subdf[1:end-1,:elapsed_time_mean])
        #log_10_2_slope = -order * log10(2) # Ideal asymptotic slope

        
        # Handle target errors that can be done by interpolation
        zipped_vecs = zip(log2_nsteps_vec, log10_relerr_vec, log10_elapsedtime_vec)
        for (i, target_error) in enumerate(target_errors)
            for ((nsteps1, relerr1, time1), (nsteps2, relerr2, time2)) in partition(zipped_vecs, 2, 1)
                if (relerr1 > target_error > relerr2)
                    target_nsteps = find_x(target_error, nsteps1, relerr1, nsteps2, relerr2)
                    target_time = find_y(target_nsteps, nsteps1, time1, nsteps2, time2)
                    target_log2nsteps_mat[i,group_i] = target_nsteps
                    target_log10time_mat[i,group_i] = target_time
                end
            end
        end

        # Use Linear Least Squares to estimate entries that can't be interpolated
        # Cut off results from before asymptotic convergence
        cutoff = findlast(x -> x > -1, log10_relerr_vec) 
        asymptotic_nsteps_vec = log2_nsteps_vec[cutoff:end] 
        asymptotic_relerr_vec = log10_relerr_vec[cutoff:end] 
        asymptotic_elapsedtime_vec = log10_elapsedtime_vec[cutoff:end]

        A = [ones(length(asymptotic_nsteps_vec)) asymptotic_nsteps_vec]

        # y = mx + b
        y_intercept_relerr, slope_relerr = A \ asymptotic_relerr_vec 
        y_intercept_time, slope_time = A \ asymptotic_elapsedtime_vec 

        for (i, target_error) in enumerate(target_errors)
            if (isnan(target_log2nsteps_mat[i,group_i])) 
                target_nsteps = (target_error - y_intercept_relerr) / slope_relerr
                target_time = y_intercept_time + slope_time*target_nsteps

                target_log2nsteps_mat[i,group_i] = target_nsteps
                target_log10time_mat[i,group_i] = target_time
            end
        end
    end

    target_nsteps_mat = round.(Int64, 2 .^ target_log2nsteps_mat)
    target_time_mat = 10.0 .^ target_log10time_mat

    integer_format(x::Integer) = format(x, commas=true)
    floatsci_format(x::Float64) = @sprintf("%.2e", x)
    float_format(x::Float64) = @sprintf("%.1f", x)
    function floatscitex_format(x::Float64)
        x_str = @sprintf("%.1e", x)
        sig, exp = split(x_str, 'e')
        exp_int = parse(Int, exp)


        return "$sig($exp_int)"
    end

    labels = String[]
    for subdf in grouped_df
        order = subdf[1,:order]
        usejuqbox = subdf[1,:usejuqbox]
    
        if usejuqbox
            if order != 2
                throw("Data processing error. usejuqbox=true, but Order≠2.")
            end
            label = "Order 2 (St\u00f6rmer-Verlet)"
        else
            label = "Order $order"
        end
        push!(labels, label)
    end

    header = hcat("Target Error", [label for _ in 1:1, label in labels])
    #first_col = string.(target_errors)
    first_col = floatscitex_format.(Float64.(10.0 .^ target_errors))

    # Speedup over 2nd order method
    target_speedup_mat = target_time_mat[:,1] ./ target_time_mat 
     
    target_nsteps_str_mat = hcat(first_col, integer_format.(target_nsteps_mat))
    target_nsteps_str_mat = vcat(header, target_nsteps_str_mat)
    target_time_str_mat = hcat(first_col, floatscitex_format.(target_time_mat))
    target_time_str_mat = vcat(header, target_time_str_mat)
    target_speedup_str_mat = hcat(first_col, float_format.(target_speedup_mat))
    target_speedup_str_mat = vcat(header, target_speedup_str_mat)

    println("Nsteps Mat")
    table(target_nsteps_str_mat)
    println("\nTime Mat")
    table(target_time_str_mat)
    println("\nSpeedup Mat")
    table(target_speedup_str_mat)

    return nothing
end

results_regex::Regex = r"""cnot3StepsizeTest
_order=(\d+)
_degree=(\d+)
_seed=(\d+)
_atol=(.*)
_rtol=(.*)
_D1=(\d+)
_time=(.*)
_nthreads=(\d+)
(?:_usejuqbox=(true|false))? # Optionally match , doesn't appear in older files
(?:_gradient=(true|false))? # Optionally match , doesn't appear in older files
(?:_gateDuration=(\d+\.\d+))? # Optionally match , doesn't appear in older files
(?:_nCavityLevels=(\d+))? # Optionally match , doesn't appear in older files
(?:_excited=(true|false))? # Optionally match , doesn't appear in older files
\.csv
"""x # 'x' tag ignores whitespace and comments

# Filename format for final states files
fstates_regex::Regex = r"""cnot3StepsizeTest
_order=(\d+)
_degree=(\d+)
_seed=(\d+)
_atol=(.*)
_rtol=(.*)
_D1=(\d+)
_time=(.*)
_nthreads=(\d+)
(?:_usejuqbox=(true|false))? # Optionally match , doesn't appear in older files
(?:_gradient=(true|false))? # Optionally match , doesn't appear in older files
(?:_gateDuration=(.*))? # Optionally match , doesn't appear in older files
(?:_nCavityLevels=(\d+))? # Optionally match , doesn't appear in older files
_finalStates
\.csv
"""x # 'x' tag ignores whitespace and comments

    

#data_directory = "./DataMar28/"
#data_directory = "./StepsizeTol1e-12/"
#data_directory = "/home/spencer/Research/QuantumGateDesign.jl/cnot3_stepsize/DataMay4/UnexcitedData"
data_directory = "/home/spencer/Research/QuantumGateDesign.jl/cnot3_stepsize/DataMay4/ExcitedData"

# Grab data from directory, combine into one data frame
results_files = filter(x -> occursin(results_regex, x),
                       readdir(data_directory, join=true))
fstates_files = filter(x -> occursin(fstates_regex, x),
                       readdir(data_directory, join=true))

results_dfs = [load_dataframe_with_metadata(file) for file in results_files]
fstates_dfs = [load_fstates_with_metadata(file) for file in fstates_files]

results_df = vcat(results_dfs...; cols=:union) # cols=:union imputes missing columns
fstates_df = vcat(fstates_dfs...; cols=:union)

common_cols = intersect(names(results_df), names(fstates_df))

# Right now, this can't handle multiple instances of the exact same test. I
# think that's fine, but it could be nice to duplicate tests for timing purposes.
full_df = innerjoin(results_df, fstates_df, on=common_cols)

group_cols = [:usejuqbox, :order, :nsteps]
processed_df = @chain full_df begin
    @groupby(group_cols) # Could also do tolerance, if needed
    @combine(:relerr_mean = mean(:relerr), :relerr_stddev = std(:relerr),
             :elapsed_time_mean = mean(:elapsed_time),
             :elapsed_time_stddev = std(:elapsed_time),
    )
end
## Something like this could automatically do all the means and stddevs
#value_cols = names(df, Not(group_cols))
#@combine gdf begin
#    (; [Symbol(c, "_mean") => mean(:($(c))) for c in value_cols]...,
#       [Symbol(c, "_std")  => std(:($(c)))  for c in value_cols]...)
#end

##### Do the plotting
