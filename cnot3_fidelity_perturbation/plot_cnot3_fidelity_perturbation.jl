using CairoMakie, LaTeXStrings, LinearAlgebra, Statistics, DataFrames, CSV, DataFramesMeta
using Makie: wong_colors, automatic
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

csv_file_pattern = r"""
cnot3PerturbationTest
_order=(\d+)
_degree=(\d+)
_seed=(\d+)
_targetErrorFine=(.+)
_targetErrorCoarse=(.+)
_nstepsFine=(\d+)
_nstepsCoarse=(\d+)
_atol=(.+)
_rtol=(.+)
_D1=(\d+)
_costType=(.+)
_gateDuration=(.+)
_nCavityLevels=(\d+)
.csv"""x # 'x' tag ignores whitespace and comments

# I have 6 dimensions: fine_seed, coarse/pert_seed, order, coarse_err,
#                      pert_order, and relative error
# Seed can be eliminated using mean + stddev
# y-axis is definitely relative error
# Order can be done using different line colors
# That leaves us with pert_order and coarse_err. One option is to have a
# different plot for each coarse error. That would be consistent with my
# optimization figure, so let's plan on that for now.
# Actually, there's two seeds. One for the fine control vector, and one for the
# coarse control vector (perturbation). But we can always add that to the mean+stddev

"""
Read csv as datafram, grab PARAM=value pairs from the filename, and add them as
rows to the dataframe.
"""
function read_csv_with_params(filepath::String)
    # Read CSV
    df = CSV.read(filepath, DataFrame, stripwhitespace=true)

    # Strip path and file extension
    filename = split(basename(filepath), ".")[1]

    # Extract parameter-value pairs
    for pair in split(filename, "_")
        if occursin("=", pair)
            param, value = split(pair, "=")
            pvalue = tryparse(Int, value) # parsed value
            pvalue = isnothing(pvalue) ? tryparse(Float64, value) : pvalue
            pvalue = isnothing(pvalue) ? value : pvalue # Default back to original string
            df = @transform(df, cols(param) = pvalue) # Add column with constant value
        end
    end

    return df
end

directory = "Apr24"

recollect = true

# Idea, just go into each file, grab data frame, insert new rows, and reduce via vcat
if recollect
    matches_csv(file) = occursin(csv_file_pattern, file)
    all_files = readdir(directory)
    matching_files = all_files[findall(matches_csv, all_files)]
    matching_files = joinpath.(directory, matching_files)

    df = mapreduce(read_csv_with_params, vcat, matching_files)
end

method_orders = sort(unique(df[:,"order"]))
coarse_errs = sort(unique(df[:,"targetErrorCoarse"]))

inch=96
fig = CairoMakie.Figure(size=(6.25inch, 5.25inch), fontsize=11, figure_padding=(0.015inch,0.05inch,0,0.025inch))

x_param = :pert_order
y_param = :objective_err
#x_param = :pert_order
#y_param = :UT_coarse_err

fig_axes = Axis[]
nrows = 2
ncols = 3
#for (i, order) in enumerate(orders)
for (i, order) in enumerate(method_orders)
    row = mod(i - 1, nrows) + 1
    col = div(i - 1, nrows) + 1
    ax = Axis(
        fig[row,col],
        title="Order $order",
        xlabel=string(x_param),
        ylabel=string(y_param),
        xscale=log10,
        yscale=log10,
    )
    push!(fig_axes, ax)
end


for (i, order) in enumerate(method_orders)
    ax = fig_axes[i]
    for (j, coarse_err) in enumerate(coarse_errs)
        reduced_df = @subset(df, :order .== order, :targetErrorCoarse .== coarse_err)
        grouped_df = @groupby(reduced_df, x_param)
        combined_df = @combine(
            grouped_df,
            :y_mean = mean(cols(y_param)),
            :y_std = std(cols(y_param)),
        )

        display(combined_df)
        
        x_axis = combined_df[:, x_param]
        line_mean   = combined_df[:, :y_mean]
        line_uband  = line_mean .+ combined_df[:, :y_std]
        line_lband  = line_mean .- combined_df[:, :y_std]

        lines!(ax, x_axis, line_mean, label="Target Error $coarse_err")
        band!(ax, x_axis, line_lband, line_uband, alpha=0.75)
    end
end

Legend(fig[end+1,:], fig_axes[1], orientation=:horizontal, framevisible=false,
       nbanks=2, tellwidth=false)
fig
