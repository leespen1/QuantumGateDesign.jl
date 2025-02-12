using DelimitedFiles, CairoMakie, LaTeXStrings
using Makie: wong_colors
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

    for line in lines[start_line:end_line]
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


txt_file_pattern = r"""
targetError=(-?[1-9](?:\.\d+)?[Ee][-+]?\d+|\d+) # Floating point regex
_cnot3OptimizationTest
_order=(\d+)
_degree=(\d+)
_seed=(\d+)
_nsteps=(\d+)
_atol=(-?[1-9](?:\.\d+)?[Ee][-+]?\d+|\d+)
_rtol=(-?[1-9](?:\.\d+)?[Ee][-+]?\d+|\d+)
_D1=(\d+)
_time=12.0
_maxiter=(\d+)
_nthreads=(\d+)
.txt"""x # 'x' tag ignores whitespace and comments

#'targetError=1e-1_cnot3OptimizationTest_order=10_degree=14_seed=3_nsteps=175_atol=1.0e-15_r
#tol=1.0e-15_D1=15_time=12.0_maxiter=10000_nthreads=4.csv'

directory = "49349033"
data = missing
header = missing
iter_vec = missing
infidelity_vec = missing

target_errors = ("1e-1", "1e-5", "1e-7")
orders = (2,4,6,8,10,12)

N_files_found = zeros(length(orders), length(target_errors))

inch = 96
#fig = CairoMakie.Figure(size=(12inch, 6inch), fontsize=12, figure_padding=5)
fig = CairoMakie.Figure(size=(6.5inch, 3.25inch), fontsize=12, figure_padding=5)
fig_axes = Axis[]
for i in eachindex(target_errors)
    if i == 1
        push!(
            fig_axes,
            Axis(
                fig[1,i], xlabel="# of IPOPT Iterations", ylabel="|Gate Infidelity|", yscale=log10,
                title="Target Error = $(target_errors[i])",
                yticks=(10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]),
                yminorticks=IntervalsBetween(10), yminorticksvisible=true,
                yminorgridvisible=false,
            )
        )
    else
        push!(
            fig_axes,
            Axis(
                fig[1,i], xlabel="# of IPOPT Iterations", yscale=log10,
                title="Target Error = $(target_errors[i])",
                yticks=(10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]), # Labels
                #yticks=(10.0 .^ (-15:15), ["" for i in -15:15]), # No labels
                yminorticks=IntervalsBetween(10), yminorticksvisible=true,
                yminorgridvisible=false,
            )
        )
    end
end

line_opacity = 0.9
line_width = 1.0

#linkyaxes!(fig_axes...)
# Dummy loop which draw empty lines, just so the Legend can be drawn
# This loop is just to make empty lines for each order, so I can draw the legend before I have hundreds of lines
for (i, order) in enumerate(orders)
    lines!(fig_axes[1], [1], [1], color=(wong_colors()[i], line_opacity), label="Order $order")
end
Legend(fig[2,:], fig_axes[1], orientation=:horizontal, framevisible=false)



iter_vecs = Vector{Int64}[]
infidelity_vecs = Vector{Float64}[]
i_target_vec = Int[]
i_order_vec = Int[]

for file in readdir(directory)
    if occursin(txt_file_pattern, file)
        regex_match = match(txt_file_pattern, file)
        target_err = regex_match[1]
        order = parse(Int, regex_match[2])

        csv_file = replace(file, ".txt" => ".csv")
        data, header = readdlm(directory * "/" * csv_file, ',', header=true)
        #@show header
        #@show file
        
        non_wdog_iter = non_watchdog_iterations(directory * "/" * file)
        non_wdog_rows = non_wdog_iter .+ 1 # Ipopt iterations are 0-indexed
        non_wdog_data = data[non_wdog_rows,:]

        header_vec = reshape(header, :)
        i_iter = findfirst(x -> x == "iter_count", header_vec)
        i_infidelity = findfirst(x -> x == "infidelity", header_vec) 

        iter_vec = non_wdog_data[:, i_iter]
        # Absolute value so log scale doesn't mess up
        infidelity_vec = abs.(non_wdog_data[:, i_infidelity])

        i_target = findfirst(x -> x == target_err, target_errors)
        i_order = findfirst(x -> x == order, orders)

        #lines!(fig_axes[i_target], iter_vec, infidelity_vec, color=(wong_colors()[i_order], line_opacity))

        push!(iter_vecs, iter_vec)
        push!(infidelity_vecs, infidelity_vec)
        push!(i_target_vec, i_target)
        push!(i_order_vec, i_order)


        N_files_found[i_order, i_target] += 1
    end
end


# Draw the lowest order lines first, highest order lines last
for desired_i_order in reverse(eachindex(orders))
#for desired_i_order in eachindex(orders)
    for (iter_vec, infidelity_vec, i_target, i_order) in zip(iter_vecs, infidelity_vecs, i_target_vec, i_order_vec)
        if i_order == desired_i_order
            lines!(fig_axes[i_target], iter_vec, infidelity_vec, color=(wong_colors()[i_order], line_opacity), linewidth=line_width)
        end
    end
end

for ax in fig_axes
    ylims!(ax, (1e-6,1e0))
end

fig
