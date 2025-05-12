using CairoMakie, LaTeXStrings, Statistics, IterTools, Printf, Format
using DataFrames, DataFramesMeta, CSV, DelimitedFiles
using LinearAlgebra
import Makie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

function non_watchdog_iterations(log_filename::String)
    iteration_numbers = Int[]

    input_string = read(log_filename, String)
    lines = split(input_string, "\n")

    # Find the portion of the file corresponding to the optimization iteration report
    #header = "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls"

    header_entries = ("iter", "objective", "inf_pr", "inf_du", "lg(mu)",
                      "||d||", "lg(rg)", "alpha_du", "alpha_pr",  "ls")

    nonzero_whitespace = raw"\s+"
    header_regex = Regex(join(header_entries, nonzero_whitespace))
    end_line_regex = r"^Number of Iterations.*"

    start_line = findfirst(line -> occursin(header_regex, line), lines)
    end_line = findlast(line -> occursin(end_line_regex, line), lines)

    # A data row has the right number of entries, and the first entry is an integer
    N_columns = length(header_entries)
    data_regex = Regex(raw"^\s*\d+" * repeat(raw"\s+\S+", N_columns-1) * raw"\s*$")

    #for line in lines[start_line:end_line]
    for line in lines[start_line:end]
        # Only do iteration lines (which start with whitespace, followed by digits)
        if occursin(data_regex, line)
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

filepath = "/home/spencer/Research/QuantumGateDesign.jl/cnot3_accuracy_verification/DataMay4/comparison_targetError=1e-1_cnot3OptimizationTest_order=4_degree=14_seed=0_nsteps=825_atol=1.0e-10_rtol=1.0e-10_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv"
#filepath = "/home/spencer/Research/QuantumGateDesign.jl/cnot3_accuracy_verification/DataMay4/comparison_targetError=1e-3_cnot3OptimizationTest_order=4_degree=14_seed=0_nsteps=3535_atol=1.0e-10_rtol=1.0e-10_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv"
df = load_dataframe_with_metadata(filepath)

### Set up Makie Figures, Axes
ticks_10f(i) = L"10^{%$i}"
log10_ticks = (10.0 .^ (-15:15), ticks_10f.(-15:15))

unlabeled_timestep_ticks = (2 .^ (5:5:20), ["" for i in 5:5:20])
labeled_timestep_ticks = (2 .^ (5:5:20), [L"2^{%$i}" for i in 5:5:20])
minor_timestep_ticks = 2 .^ (0:20)
inch = 96 # Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
fig = CairoMakie.Figure(size=(5.75inch, 2.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))

nsteps_vs_error_grid = fig[1,1]
nsteps_vs_time_grid = fig[2,1]
error_vs_time_grid = fig[1:2,2]

ax = CairoMakie.Axis(
    fig[1,1],
    yscale=CairoMakie.log10,
    #xlabel="Number of Timesteps",
    xlabel="Ipopt Iteration #",
    xminorticksvisible = true,
    xminorgridvisible = true,
    yticks=log10_ticks,
    yminorticks=IntervalsBetween(10),
    yminorticksvisible=true,
    yminorgridvisible=true,
    limits=((0,nothing), (1e-6, 1e0)),
)

iter = 1 .+ non_watchdog_iterations(replace(filepath, "_pcof.csv" => ".txt", "comparison_" => ""))
df = df[iter,:]

lines!(ax, df[:,:ipopt_iter], df[:,:coarse_gen_infidelity], label="Generalized Infidelity: Low-Accuracy Numerical Solution")
lines!(ax, df[:,:ipopt_iter], df[:,:fine_gen_infidelity], label="Generalized Infidelity: High-Accuracy Numerical Solution")
lines!(ax, df[:,:ipopt_iter], df[:,:UT_err] ./ df[:,:fine_norm_UT], label="Final State Relative Error")
#lines!(ax, df[:,:ipopt_iter], df[:,:gen_infidelity_err], label="Generalized Infidelity Error")

Legend(
    fig[end+1,:],
    ax,
    orientation = :horizontal,
    tellwidth = false,
    nbanks=2,
    framevisible=false,
)
rowgap!(fig.layout, 1, 0inch)

fig
