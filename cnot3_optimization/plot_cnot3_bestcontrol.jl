using DelimitedFiles, CairoMakie, LaTeXStrings, QuantumGateDesign, LinearAlgebra
using CairoMakie.Makie: wong_colors, automatic
CairoMakie.set_theme!(CairoMakie.theme_latexfonts())

csv_file_pattern = r"""
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
.csv"""x # 'x' tag ignores whitespace and comments


directory = "63397133"
target_err = "1e-7"
orders = (2,4,6,8,10,12)
objective_type = "generalized_infidelity"

x_vecs = Vector{Float64}[]
objective_vecs = Vector{Float64}[]
i_target_vec = Int[]
i_order_vec = Int[]


recollect = true
if recollect
    min_objective = Inf
    min_obj_pcof = missing
    min_obj_file = missing
    # Get the best control vector
    for file in readdir(directory)
        if occursin(csv_file_pattern, file)
            regex_match = match(csv_file_pattern, file)
            target_err_regex = regex_match[1]
            order_regex = parse(Int, regex_match[2])

            if !(order_regex in orders) || (target_err_regex != target_err)
                continue
            end

            data, header = readdlm(directory * "/" * file, ',', header=true)
            
            header_vec = reshape(header, :)
            i_objective = findfirst(x -> x == objective_type, header_vec) 
            objective_vec = data[:, i_objective]
            if objective_type == "infidelity"
                # Because the infidelity can go negative due to numerical error
                objective_vec = abs.(objective_vec)
            end


            this_min_obj, min_obj_i = findmin(objective_vec)
            if this_min_obj < min_objective
                global min_objective = this_min_obj
                global min_obj_file = file

                pcof_file = replace(file, ".csv" => "_pcof.csv")
                pcofs = readdlm(directory * "/" * pcof_file, ',', Float64)
                pcof = pcofs[min_obj_i,:]

                global min_obj_pcof = pcof
            end
        end
    end

    println("Best optimization result: ", min_obj_file)

    min_obj_rgx = match(csv_file_pattern, min_obj_file)

    order = parse(Int, min_obj_rgx[2])
    degree = parse(Int, min_obj_rgx[3])
    seed = parse(Int, min_obj_rgx[4])
    nsteps = parse(Int, min_obj_rgx[5])
    atol = parse(Float64, min_obj_rgx[6])
    rtol = parse(Float64, min_obj_rgx[7])
    D1 = parse(Int, min_obj_rgx[8])
    gateDuration = parse(Float64, min_obj_rgx[13])
    nCavityLevels = parse(Int, min_obj_rgx[14])
    cnot3ret = QuantumGateDesign.setup_cnot3(
        seed=seed, # This shouldn't matter
        atol=atol,
        rtol=rtol,
        D1=D1,
        N_osc_levels=nCavityLevels,
        Tmax=gateDuration
    )
    cnot3ret.qgd_prob.nsteps = nsteps
    prob = cnot3ret.qgd_prob
    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    guard_weights = diag(cnot3ret.juqbox_params.wmat)

    history = eval_forward(prob, controls, min_obj_pcof, order=order)
    population_history = abs2.(history)

    ts = LinRange(0, gateDuration, 1+nsteps)
    control_history = fill(NaN+NaN*im, length(controls), 1+nsteps)
    for (control_i, control) in enumerate(controls)
        local_pcof = QuantumGateDesign.get_control_vector_slice(min_obj_pcof, controls, control_i)
        for (t_i, t) in enumerate(ts)
            p = eval_p_derivative(control, t, local_pcof, 0)
            q = eval_q_derivative(control, t, local_pcof, 0)
            control_history[control_i, t_i] = p + q*im
        end
    end
end # recollect


initial_cond = 4
i_00 = findfirst(isequal(1), history[:,1,1])
i_01 = findfirst(isequal(1), history[:,1,2])
i_10 = findfirst(isequal(1), history[:,1,3])
i_11 = findfirst(isequal(1), history[:,1,4])
essential_levels = [i_00, i_01, i_10, i_11]
subsys_sizes = (10,4,4)
labels = ["|00⟩","|01⟩","|10⟩","|11⟩"]
xlims = (0, gateDuration)
ylims = (0,1)

inch = 96 # Getting correct figure, font size: https://docs.makie.org/stable/how-to/match-figure-size-font-sizes-and-dpi
fig = CairoMakie.Figure(size=(6.25inch, 5.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))
#fig = CairoMakie.Figure(size=(6.25inch, 10.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))

fig_axes = Axis[]
axis_fig_locations = ((1,1), (1,2), (2,1), (2,2)) # Grid
#axis_fig_locations = ((1,1), (2,1), (3,1), (4,1)) # Vertical Stack

# Get labels for the guard states with significant population. Need to do this
# now so that the labeling is consistent across each subfigure
significant_guard_levels = Int64[]
for n in 1:size(population_history, 1)
    if !(n in essential_levels) && (maximum(population_history[n,:,:]) > 0.01)
        push!(significant_guard_levels, n)
    end
end

titles = [L"|000\rangle \rightarrow |000\rangle",
          L"|001\rangle \rightarrow |001\rangle",
          L"|010\rangle \rightarrow |011\rangle",
          L"|011\rangle \rightarrow |010\rangle",
         ]

for (initial_cond, fig_loc) in enumerate(axis_fig_locations)

    xlabel = fig_loc[1] == maximum(first, axis_fig_locations) ? "Time (nanoseconds)" : ""
    ylabel = fig_loc[2] == minimum(last, axis_fig_locations) ? "Population" : ""
    xticklabelsvisible = fig_loc[1] == maximum(first, axis_fig_locations) 
    yticklabelsvisible = fig_loc[2] == minimum(first, axis_fig_locations) 
    #ylabel = fig_loc[1] minimum(last, axis_fig_locations) ? "Population" : ""
    #xticklabelsvisible = fig_loc[1] == 2 ? true : false
    #yticklabelsvisible = fig_loc[2] == 1 ? true : false
    
    ax = Axis(
        fig[fig_loc...],
        title=titles[initial_cond],
        xlabel = xlabel,
        ylabel = ylabel,
        yticks = 0:0.25:1,
        yticklabelsvisible = yticklabelsvisible,
        yminorticks = IntervalsBetween(2),
        yminorticksvisible = true,
        xticks = 0:100:600,
        xticklabelsvisible = xticklabelsvisible,
        xminorticks = IntervalsBetween(2),
        xminorticksvisible = true,
        limits = (xlims, ylims)
    )
    push!(fig_axes, ax)


    # Essential states
    for level in essential_levels
        state_population_hist = population_history[level, :, initial_cond]
        #lines!(ax, ts, state_population_hist, label=labels[n])
        lines!(ax, ts, state_population_hist,
               label=basis_state_to_string(index_to_basis_state(level, subsys_sizes))
        )
    end


    # Total Guard Population, 
    sum_guard_populations = zeros(1+nsteps)
    sum_essential_populations = zeros(1+nsteps)
    state_lengths = [norm(x) for x in eachcol(history[:,:,initial_cond])]
    for n in 1:size(history,1)
        state_population_hist = population_history[n, :, initial_cond]
        if n in essential_levels
            sum_essential_populations .+= state_population_hist
        else
            sum_guard_populations .+= state_population_hist
        end
    end
    lines!(ax, ts, sum_guard_populations, label="Guard Population")
    #lines!(ax, ts, sum_essential_populations, label="Essential Population")
    #lines!(ax, ts, state_lengths, label="State Vector Length")
    
    #colors = Makie.wong_colors()
    #for level in significant_guard_levels
    #    state_population_hist = population_history[level, :, initial_cond]
    #    lines!(ax, ts, state_population_hist,
    #           label=basis_state_to_string(index_to_basis_state(level, subsys_sizes)),
    #           linestyle=:dot, color=colors[1 + (level % length(colors))]
    #    )
    #end
    
    # Handle |(1-5),(0-1),(0-1)⟩, |(0-5),2,(0-1)⟩, |(0-5),(0-1),2)⟩, and |(0-5),2,2)⟩
    # This way I only need four more lines
    sum_guard_pop1 = zeros(1+nsteps)
    sum_guard_pop2 = zeros(1+nsteps)
    sum_guard_pop3 = zeros(1+nsteps)
    sum_guard_pop4 = zeros(1+nsteps)
    sum_guard_pop_higher = zeros(1+nsteps)
    sum_guard_pop_forbidden = zeros(1+nsteps) # Guard states at the 'edge' of our model
    resonator_guard_cutoff = 4
    for level in 1:size(history,1)
        state_population_hist = population_history[level, :, initial_cond]
        nR, n1, n2 = index_to_basis_state(level, subsys_sizes)
        if (nR in 1:resonator_guard_cutoff) && (n1 in 0:1) && (n2 in 0:1)
            sum_guard_pop1 += state_population_hist
        elseif (nR in 0:resonator_guard_cutoff) && (n1 in 0:1) && (n2 == 2)
            sum_guard_pop2 += state_population_hist
        elseif (nR in 0:resonator_guard_cutoff) && (n1 == 2) && (n2 in 0:1)
            sum_guard_pop3 += state_population_hist
        elseif (nR in 0:resonator_guard_cutoff) && (n1 == 2) && (n2 == 2)
            sum_guard_pop4 += state_population_hist
        elseif !(level in essential_levels)
            sum_guard_pop_higher += state_population_hist
        end

        if (n1 == 3) || (n2 == 3) || (nR == 9)
            sum_guard_pop_forbidden += state_population_hist
        end
    end
    #set_theme!(color = :auto) # Reset color cycle
    #ax.cycler.counters[Lines]
    #my_theme = Theme()
    #with_theme(
    #    Theme(
    #        palette = (color = [:red, :blue], linestyle = [:dash, :dot]),
    #        Lines = (cycle = Cycle([:color, :linestyle], covary = true),)
    #    )) do
    #    lines!(ax, ts, sum_guard_pop1, linestyle=:dot, label="|1-5,0-1,0-1⟩")
    #    lines!(ax, ts, sum_guard_pop2, linestyle=:dot, label="|0-5,0-1,2⟩")
    #    lines!(ax, ts, sum_guard_pop3, linestyle=:dot, label="|0-5,2,0-1⟩")
    #    lines!(ax, ts, sum_guard_pop4, linestyle=:dot, label="|0-5,2,2⟩")
    #    lines!(ax, ts, sum_guard_pop5, linestyle=:dot, label="Other Guard Levels")
    #end
    lines!(ax, ts, sum_guard_pop1, color=Cycled(1), linestyle=:dash, label="|1-4,0-1,0-1⟩")
    lines!(ax, ts, sum_guard_pop2, color=Cycled(2), linestyle=:dash, label="|0-4,0-1,2⟩")
    lines!(ax, ts, sum_guard_pop3, color=Cycled(3), linestyle=:dash, label="|0-4,2,0-1⟩")
    lines!(ax, ts, sum_guard_pop4, color=Cycled(4), linestyle=:dash, label="|0-4,2,2⟩")
    lines!(ax, ts, sum_guard_pop_higher, color=Cycled(5), linestyle=:dash, label="Higher Guard Levels")
    @show  norm(sum_guard_pop_forbidden, Inf)
end

# Plot the controls

Legend(fig[end+1,:], fig_axes[1], orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)

control_ax = Axis(
    fig[end+1,:],
    title= "Control Pulses",
    xlabel = "Time (nanoseconds)",
    ylabel = "Amplitude (MHz)",
    #yticks = 0:0.25:1,
    #yticklabelsvisible = yticklabelsvisible,
    #yminorticks = IntervalsBetween(2),
    #yminorticksvisible = true,
    xticks = 0:100:600,
    xticklabelsvisible = true,
    xminorticks = IntervalsBetween(2),
    xminorticksvisible = true,
    limits = ((0,550), nothing)
)

for control_i in 1:length(controls)
    ps = real(control_history[control_i,:])
    qs = imag(control_history[control_i,:])
    
    ps .*= 1_000 # Convert from GHz to MHz
    qs .*= 1_000

    control_label_subscript = control_i == 3 ? 'R' : control_i

    lines!(control_ax, ts, ps, label=L"\textrm{Re } c_%$(control_label_subscript)")
    lines!(control_ax, ts, qs, label=L"\textrm{Im } c_%$(control_label_subscript)")
end

Label(fig[0, :], "Time Evolution of State Populations", halign = :center, font=:bold)
Legend(fig[end+1,:], control_ax, orientation = :horizontal, tellwidth = false, nbanks=1, framevisible=false)

rowgap!(fig.layout, 1, 0.0inch)

rowgap!(fig.layout, 2, 0.1inch)
rowgap!(fig.layout, 3, 0.0inch)
colgap!(fig.layout, 1, 0.1inch)

rowgap!(fig.layout, 4, 0.0inch)
rowgap!(fig.layout, 5, 0.0inch)

fig



## Now handle the real and imaginary part plotting for one iniitial condition
# Essential states (real and imaginary part)
fig_realimag = CairoMakie.Figure(size=(5.25inch, 4.5inch), fontsize=11, figure_padding=(0.015inch,0.15inch,0.0inch,0.075inch))
ax_real = Axis(
    fig_realimag[1,1],
    title=L"\textbf{Time Evolution of Probability Amplitudes, } |\psi_0\rangle = |011\rangle",
    #xlabel = xlabel,
    ylabel = "Real Part",
    yticks = -1:0.5:1,
    yticklabelsvisible = true,
    yminorticks = IntervalsBetween(2),
    yminorticksvisible = true,
    xticks = 0:100:600,
    xticklabelsvisible = false,
    xminorticks = IntervalsBetween(2),
    xminorticksvisible = true,
    limits = (xlims, (-1, 1))
)
ax_imag = Axis(
    fig_realimag[2,1],
    #title="Imaginary Part",
    xlabel = "Time (nanoseconds)",
    ylabel = "Imaginary Part",
    yticks = -1:0.5:1,
    yticklabelsvisible = true,
    yminorticks = IntervalsBetween(2),
    yminorticksvisible = true,
    xticks = 0:100:600,
    xticklabelsvisible = true,
    xminorticks = IntervalsBetween(2),
    xminorticksvisible = true,
    limits = (xlims, (-1, 1))
)

# Use the highest-energy initial condition.
real_imag_init_cond = 4

# Get the 10 levels with the highest population
total_populations = sum(population_history[:,:,real_imag_init_cond], dims=2) |> vec
num_top_levels = 6
top_pop_levels = sortperm(total_populations, rev=true)[1:num_top_levels]

#for level in essential_levels
#for level in 1:size(history, 1)
    #nR, n1, n2 = index_to_basis_state(level, subsys_sizes)
    #if (n1 == 3) || (n2 == 3) || (nR == 9)
    #    continue # Skip "Forbidden" states
    #end
for level in top_pop_levels

    state_hist = history[level, :, real_imag_init_cond]
    #lines!(ax, ts, state_population_hist, label=labels[n])
    lines!(ax_real, ts, real(state_hist), linewidth=1,
           label=basis_state_to_string(index_to_basis_state(level, subsys_sizes))
    )
    lines!(ax_imag, ts, imag(state_hist), linewidth=1,
           label=basis_state_to_string(index_to_basis_state(level, subsys_sizes))
    )
end
Legend(fig_realimag[end+1,:], ax_real, orientation = :horizontal, tellwidth = false, nbanks=1, framevisible=false)

fig_realimag

