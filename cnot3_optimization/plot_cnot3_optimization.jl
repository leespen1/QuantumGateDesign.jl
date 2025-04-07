using DelimitedFiles, CairoMakie, LaTeXStrings
using Makie: wong_colors, automatic
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
_time=(.+)
_maxiter=(\d+)
_nthreads=(\d+)
_costType=(.+)
_gateDuration=(.+)
_nCavityLevels=(\d+)
.txt"""x # 'x' tag ignores whitespace and comments


directory = "51458828"
target_errors = ("1e-1", "1e-3", "1e-5", "1e-7")
target_errors_title = ("10^{-1}", "10^{-3}", "10^{-5}", "10^{-7}")
orders = (2,4,6,8,10,12)
xaxis = "iter_count"
objective_type = "generalized_infidelity"
line_opacity = 0.9
line_width = 1.0
inch = 96
fig = CairoMakie.Figure(size=(6.25inch, 3.25inch), fontsize=11, figure_padding=(0.015inch,0.05inch,0,0.025inch))

if xaxis == "elapsed_time"
    xlabel = "Wall Time Elapsed (Hours)"
    xticks = 0:6
    xlims = (0,6)
elseif xaxis == "iter_count"
    xlabel = "Number of IPOPT Iterations Completed"
    xticks = automatic
    xlims = (0,nothing)
else
    xlabel = "Unknown"
    xticks = automatic
    xlims = nothing
end

ylims = (1e-7,1e0)
xminorticks = IntervalsBetween(2)

labeled_yticks =(10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15])
unlabeled_yticks =(10.0 .^ (-15:15), ["" for i in -15:15])

if objective_type == "generalized_infidelity"
    ylabel = "Generalized Gate Infidelity"
elseif objective_type == "infidelity"
    ylabel = "Gate Infidelity"
else 
    ylabel = "Unknown"
end


fig_axes = Axis[]
for i in eachindex(target_errors)
    this_ylabel = (i == 1) ? ylabel : ""
    this_yticks = (i == 1) ? labeled_yticks : unlabeled_yticks

    push!(
        fig_axes,
        Axis(
            fig[1,i], 
            title=L"\textrm{Target Error} = %$(target_errors_title[i])",
            ylabel=this_ylabel, 
            yscale=log10,
            yticks=this_yticks,
            yminorticks=IntervalsBetween(10),
            yminorticksvisible=true,
            yminorgridvisible=false,
            xticks = xticks,
            xminorticks = xminorticks,
            xminorticksvisible=true,
            limits = (xlims, ylims),
        )
    )
end


#linkyaxes!(fig_axes...)
# Dummy loop which draw empty lines, just so the Legend can be drawn
for (i, order) in enumerate(orders)
    lines!(fig_axes[1], [1], [1], color=(wong_colors()[i], line_opacity), label="Order $order")
end



N_files_found = zeros(Int64, length(orders), length(target_errors))
x_vecs = Vector{Float64}[]
objective_vecs = Vector{Float64}[]
i_target_vec = Int[]
i_order_vec = Int[]

for file in readdir(directory)
    if occursin(txt_file_pattern, file)
        regex_match = match(txt_file_pattern, file)
        target_err = regex_match[1]
        order = parse(Int, regex_match[2])

        csv_file = replace(file, ".txt" => ".csv")
        data, header = readdlm(directory * "/" * csv_file, ',', header=true)
        
        # Remove watchdog iterations
        non_wdog_iter = non_watchdog_iterations(directory * "/" * file)
        non_wdog_rows = non_wdog_iter .+ 1 # Ipopt iterations are 0-indexed
        non_wdog_data = data[non_wdog_rows,:]

        header_vec = reshape(header, :)
        i_x = findfirst(x -> x == xaxis, header_vec)
        i_objective = findfirst(x -> x == objective_type, header_vec) 

        x_vec = non_wdog_data[:, i_x]
        objective_vec = non_wdog_data[:, i_objective]
        if xaxis == "elapsed_time"
            x_vec ./= 3600 # Convert seconds to hours
        end
        if objective_type == "infidelity"
            # Because the infidelity can go negative due to numerical error
            objective_vec = abs.(objective_vec)
        end

        i_target = findfirst(x -> x == target_err, target_errors)
        i_order = findfirst(x -> x == order, orders)

        #lines!(fig_axes[i_target], iter_vec, objective_vec, color=(wong_colors()[i_order], line_opacity))

        push!(x_vecs, convert(Vector{Float64}, x_vec))
        push!(objective_vecs, objective_vec)
        push!(i_target_vec, i_target)
        push!(i_order_vec, i_order)

        N_files_found[i_order, i_target] += 1
    end
end
@show N_files_found


# Draw the lowest order lines first, highest order lines last
#for desired_i_order in reverse(eachindex(orders))
for desired_i_order in eachindex(orders)
    for (x_vec, objective_vec, i_target, i_order) in zip(x_vecs, objective_vecs, i_target_vec, i_order_vec)
        if i_order == desired_i_order
            lines!(fig_axes[i_target], x_vec, objective_vec, color=(wong_colors()[i_order], line_opacity), linewidth=line_width)
        end
    end
end

#for ax in fig_axes
#    #ylims!(ax, (1e-6,1e0))
#    ylims!(ax, (1e-7,1e0))
#end

Label(fig[2, :], xlabel, valign=:top)
Legend(fig[3,:], fig_axes[1], orientation=:horizontal, framevisible=false, tellwidth=false)

rowgap!(fig.layout, 1, 0.1inch)
rowgap!(fig.layout, 2, 0.0inch)
colgap!(fig.layout, 1, 0.1inch)
colgap!(fig.layout, 2, 0.1inch)
colgap!(fig.layout, 3, 0.1inch)

fig
