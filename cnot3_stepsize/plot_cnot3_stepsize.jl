using Plots, DelimitedFiles, LaTeXStrings
import Makie
import CairoMakie
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

function get_data(x_header, y_header, out_order, data_directory=missing)
    file_pattern= r"""cnot3StepsizeTest
    _order=(\d+)
    _degree=(\d+)
    _seed=(\d+)
    _atol=(.*)
    _rtol=(.*)
    _D1=(\d+)
    _time=(.*)
    _nthreads=(\d+)
    .csv
    """x # 'x' tag ignores whitespace and comments


    top_dict = Dict(2 => Dict(), 4 => Dict(), 6 => Dict(), 8 => Dict(), 10 => Dict(), 12 => Dict())

    x_data_entries = Any[]
    y_data_entries = Any[]

    if ismissing(data_directory)
        data_directory = dirname(@__FILE__) * "/Data/"
    end

    files_found = 0
    files = readdir(data_directory)
    for file in files
        if occursin(file_pattern, file)
            regex_match = match(file_pattern, file)
            order    = parse(Int,     regex_match[1])
            degree   = parse(Int,     regex_match[2])
            seed     = parse(Int,     regex_match[3])
            atol     = parse(Float64, regex_match[4])
            rtol     = parse(Float64, regex_match[5])
            D1       = parse(Int,     regex_match[6])
            time     = parse(Float64, regex_match[7])
            nthreads = parse(Int,     regex_match[8])

            if order == out_order
                files_found += 1
                filepath = data_directory * "/" * file

                data, header = readdlm(filepath, ',', header=true)
                header = vec(header)
               
                x_index = findfirst(x -> x == x_header, header)
                y_index = findfirst(x -> x == y_header, header)

                x_data = data[:, x_index]
                y_data = data[:, y_index]

                push!(x_data_entries, x_data)
                push!(y_data_entries, y_data)
            end
        end
    end

    if files_found == 0
        @warn "No files found matching conditions!"
    end

    return x_data_entries, y_data_entries

end

function get_x_vec_y_mat(x_data_entries, y_data_entries)
    if length(x_data_entries) == 0
        @warn "Length of data entries is zero, returning empty vectors and matrices"
        return zeros(0), zeros(0,0), zeros(0,0)
    end

    x_data_lengths = length.(x_data_entries)
    max_length = maximum(x_data_lengths)
    min_length = minimum(x_data_lengths)
    if max_length != min_length 
        @warn "Not all x_data entries are the same length. Max is $max_length, min is $min_length."
    end

    # Use the longest one
    x_vec = argmax(length, x_data_entries)

    n_entries = length(x_data_entries)
    x_mat = fill(NaN, max_length, n_entries)
    y_mat = fill(NaN, max_length, n_entries)
    for i in 1:n_entries
        x_data = x_data_entries[i]
        y_data = y_data_entries[i]
        l = length(y_data)

        x_mat[1:l,i] .= x_data
        y_mat[1:l,i] .= y_data
    end

    return x_vec, x_mat, y_mat
end


#data_directory = "Data"
data_directory = "48854734"
x_vecs = Any[]
y_mats = Any[]
stddev_pls = Any[]
spaghetti_pls = Any[]
makie_stddev = Any[]
makie_spaghetti = Any[]

my_xticks = 2 .^ (0:20)
my_xticklabels = [(i % 5 == 0) || (i == 2) ? L"2^{%$i}" : "" for i in 0:20]

fig3 = CairoMakie.Figure()
ax3 = CairoMakie.Axis(
    fig3[1,1],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Relative Error",
    xticks = (my_xticks, my_xticklabels),
    yticks = (10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]),
    title="Combined Orders",
)
ax4 = CairoMakie.Axis(
    fig3[2,1],
    xscale=CairoMakie.log2,
    yscale=CairoMakie.log10,
    xlabel="Number of Timesteps",
    ylabel="Elapsed Time (s)",
    xticks = (my_xticks, my_xticklabels),
    yticks = (10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]),
    title="Combined Orders",
)

orders = (2,4,6,8,10,12)
#orders = (12,)
for (k, order) in enumerate(orders)
    local x_data_entries, y_data_entries = get_data("nsteps", "R_rel_err_L2", order, data_directory)
    local x_data_entries2, y_data_entries2 = get_data("nsteps", "elapsed_time", order, data_directory)

    if length(x_data_entries) == 0
        push!(x_vecs, missing)
        push!(y_mats, missing)
        push!(stddev_pls, missing)
        push!(spaghetti_pls, missing)
        push!(makie_stddev, missing)
        push!(makie_spaghetti, missing)
        continue
    end

    local x_vec, x_mat, y_mat = get_x_vec_y_mat(x_data_entries, y_data_entries)
    local y_mean = Plots.mean(y_mat, dims=2) |> vec
    local y_std = Plots.std(y_mat, dims=2) |> vec

    push!(x_vecs, x_vec)
    push!(y_mats, y_mat)

    local stddev_pl = plot(
        x_vec, y_mean,
        ribbon=y_std,  # Adds error bars as ribbons
        label="Mean ± Std Dev",
        xlabel="Stepsize",
        ylabel="Relative Error",
        title="Mean and Standard Deviation Plot, Order=$order",
        color=:blue,
        yscale=:log10,
        xscale=:log10,
        xticks=10.0 .^ (-10:10),
        yticks=10.0 .^ (-10:10),
    )

    max_lines = min(10, size(x_mat,2))
    local spaghetti_pl = plot(
        x_mat[:,1:max_lines], y_mat[:,1:max_lines],
        label="",
        xlabel="Stepsize",
        ylabel="Relative Error",
        title="Spaghetii Plot, Order=$order",
        color=:blue,
        yscale=:log10,
        xscale=:log10,
        xticks=10.0 .^ (-10:10),
        yticks=10.0 .^ (-10:10),
        marker=:x
    )

    push!(spaghetti_pls, spaghetti_pl)
    push!(stddev_pls, stddev_pl)

    println("Order=",order)
    display(hcat(y_mean,y_std))

    local fig1 = CairoMakie.Figure()
    local ax1 = CairoMakie.Axis(
        fig1[1,1],
        xscale=CairoMakie.log10,
        yscale=CairoMakie.log10,
        xlabel="Step Size (ns)",
        ylabel="Relative Error",
        yticks = (10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]),
        title="Order = $order",
    )
    @show x_vec
    @show y_mean
    CairoMakie.lines!(ax1, x_vec, y_mean; color = :dodgerblue, label = "blue")
    CairoMakie.band!(ax1, x_vec, y_mean .- y_std, y_mean .+ y_std; color = (:dodgerblue, 0.35), label = "blue")

    push!(makie_stddev, fig1)


    local fig2 = CairoMakie.Figure()
    local ax2 = CairoMakie.Axis(
        fig2[1,1],
        xscale=CairoMakie.log2,
        yscale=CairoMakie.log10,
        xlabel="Number of Timesteps",
        ylabel="Relative Error",
        xticks = (my_xticks, my_xticklabels),
        yticks = (10.0 .^ (-15:15), [L"10^{%$i}" for i in -15:15]),
        title="Order = $order",
    )
    for i in 1:length(x_data_entries)
        CairoMakie.lines!(ax2, x_data_entries[i], y_data_entries[i]; color = (:dodgerblue, 0.6))
        CairoMakie.lines!(ax3, x_data_entries[i], y_data_entries[i]; color=(Makie.wong_colors()[k], 0.5))
        CairoMakie.lines!(ax4, x_data_entries2[i], y_data_entries2[i]; color=(Makie.wong_colors()[k], 0.5))
    end
    #CairoMakie.lines!(ax2, x_vec, y_mean, color=:red, label="Mean")

    push!(makie_spaghetti, fig2)
#ylims!(-0.55,1)
#axislegend(ax, position = :rt, merge = true)
#hidedecorations!(ax; grid = false)
#fig

end
