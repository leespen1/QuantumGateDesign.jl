module ControlVisualizer

println("Loading ControlVisualizer")

using QuantumGateDesign
using GLMakie


"""
Visualize a Control Using Makie

This is really old. From back when I was using functions as parameters for the
object instead of methods for the type. Needs to be updated to work.
"""
function QuantumGateDesign.visualize_control(controls; n_points=101, prob=missing, 
        pcof_init=missing, use_tboxes=true, target=missing)

    if !ismissing(prob)
        if prob.N_initial_conditions == 2
            labels=["|0⟩", "|1⟩"]
        elseif prob.N_initial_conditions == 4
            labels=["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
        else
            labels = []
        end
        axes_positions = [(1,1), (1,2), (2,1), (2,2)]
    end


    tf = controls[1].tf

    t_range_control = LinRange(0, tf, n_points)

    # Set up Makie figure and axis for plotting the control
    fig = Figure(;)
    ax = Axis(fig[1,1], xlabel="Time (ns)", ylabel="Control Amplitude (MHz/2π)")
    xlims!(ax, -(1/16)*tf, tf*(1+(1/16))) # Put a little bit of padding outside the control range

    #==================================
    # Set up control vector slidergrid
    ==================================#

    # Set up control vector pcof, and slidergrid for manipulating control vector
    pcof_obsv = Vector{Observable{Float64}}(undef, get_number_of_control_parameters(controls))

    # Set initial values for control vector. Initialize to user-specified values, if given
    startvalues = zeros(get_number_of_control_parameters(controls))
    if !ismissing(pcof_init)
        startvalues .= pcof_init
    end

    #pcof_range = LinRange(0, 1e-3, 11)
    pcof_range = LinRange(0, 1, 101)

    pcof_slider_parameters = [
        (label="pcof[$i]", range=pcof_range, startvalue=startvalues[i])
        for i in 1:get_number_of_control_parameters(controls)
    ]

    pcof_slidergrid = SliderGrid(fig[2,1], pcof_slider_parameters...)

    # Set up link between slider values and pcof observable values
    for i in 1:get_number_of_control_parameters(controls)
        pcof_obsv[i] = lift(x -> x, pcof_slidergrid.sliders[i].value)
    end

    # Set up function value observables
    p_vals_whole_obsv = [Observable([Point2(t_range_control[k], eval_p_single(controls, t_range_control[k], startvalues, i)*1e3/(2pi)) for k in 1:n_points]) for i in 1:length(controls)]
    q_vals_whole_obsv = [Observable([Point2(t_range_control[k], eval_q_single(controls, t_range_control[k], startvalues, i)*1e3/(2pi)) for k in 1:n_points]) for i in 1:length(controls)]

    p_vals_obsv = Observable([Point2(t_range_control[k], eval_p_single(controls, t_range_control[k], startvalues, 1)*1e3/(2pi)) for k in 1:n_points])
    q_vals_obsv = Observable([Point2(t_range_control[k], eval_q_single(controls, t_range_control[k], startvalues, 1)*1e3/(2pi)) for k in 1:n_points])

    # If prob is provided, also plot population evolution
    if !ismissing(prob)
        @assert tf == prob.tf # Make sure problem and control have same range
        t_range_prob = LinRange(0, tf, 1+prob.nsteps)

        # Set up plot for populations
        populations_grid = GridLayout()
        fig[3:5,1] = populations_grid
        population_axes = [Axis(populations_grid[axes_positions[i]...], xlabel="Time (ns)", ylabel="Populations", title="$(labels[i])") 
                           for i in 1:prob.N_initial_conditions]
        for i in 1:prob.N_initial_conditions
            xlims!(population_axes[i], -(1/16)*tf, tf*(1+(1/16)))
        end

        # Set up population value observables
        # Each initial condition has it's own population array
        populations_obsv_list = [Observable(zeros(size(prob.u0, 1), 1+prob.nsteps)) for i in 1:prob.N_initial_conditions]
        infidelity_obsv = Observable(0.0)
        infidelity_str_obsv = Observable("")

        #fig[4,1] = Label(infidelity_str_obsv)
        #infidelity_label = Label(fig[3,1], "Hello World")
        #fig[4,1] = Label("Hello World")
    end

    max_control_amp = 1.0
    min_control_amp = -1.0
    #=================================================
    # Function for updating graph if parameters change
    =================================================#
    function update_graph()
        pcof = to_value.(pcof_obsv)
        # (in units of MHz, non angular)
        for i in 1:length(controls)
            p_vals_whole_obsv[i][] = [Point2(t_range_control[k], eval_p_single(controls, t_range_control[k], pcof, i)*1e3/(2pi)) for k in 1:n_points]
            q_vals_whole_obsv[i][] = [Point2(t_range_control[k], eval_q_single(controls, t_range_control[k], pcof, i)*1e3/(2pi)) for k in 1:n_points]
        end

        p_vals_obsv[] = [Point2(t_range_control[k], eval_p_single(controls, t_range_control[k], pcof, 1) * 1e3/(2pi)) for k in 1:n_points]
        q_vals_obsv[] = [Point2(t_range_control[k], eval_q_single(controls, t_range_control[k], pcof, 1) * 1e3/(2pi)) for k in 1:n_points]

        for i in 1:length(controls)
            max_p = maximum(getindex.(p_vals_whole_obsv[i][], 2))
            max_q = maximum(getindex.(q_vals_whole_obsv[i][], 2))
            min_p = minimum(getindex.(p_vals_whole_obsv[i][], 2))
            min_q = minimum(getindex.(q_vals_whole_obsv[i][], 2))
            max_control_amp = max(max_control_amp, 1.25*max_p, 1.25*max_q)
            min_control_amp = min(min_control_amp, min_p, min_q)
        end

        # Update ylims to capture entire control; const args ensure ylims != (0,0)
        ylims!(ax, min_control_amp, max_control_amp)

        if !ismissing(prob)
            history = eval_forward(prob, controls, pcof, order=4)
            populations = get_populations(history)
            for i in 1:size(prob.u0, 2)
                populations_obsv_list[i][] = populations[:,:,i]
            end

            infidelity_obsv[] = infidelity(prob, controls, pcof, target, order=4)
            infidelity_str_obsv[] = "Fidelity: $(1 - infidelity_obsv[])"
        end
    end

    # Initial update
    update_graph()


    # Set up link so that graph updates when any control parameter changes
    for i in 1:get_number_of_control_parameters(controls)
        on(pcof_obsv[i]) do coeff
            update_graph()
        end
    end

    ## Draw the control functions
    #GLMakie.lines!(ax, p_vals_obsv, linewidth=2)
    #GLMakie.lines!(ax, q_vals_obsv, linewidth=2)
    for i in 1:length(p_vals_whole_obsv)
        GLMakie.lines!(ax, p_vals_whole_obsv[i], linewidth=2)
        GLMakie.lines!(ax, q_vals_whole_obsv[i], linewidth=2)
    end

    # Draw the populations
    if !ismissing(prob)
        #for i in 1:size(prob.u0, 2)
        for i in 1:prob.N_initial_conditions
            GLMakie.series!(population_axes[i], t_range_prob, populations_obsv_list[i], labels=labels)
        end
        #populations_grid[end,1+prob.N_initial_conditions] = Legend(fig, population_axes[end], "State")
        axislegend(population_axes[1])
    end


    ## For changing layouts. Ideally, I should have functions for constructing
    ## the various layouts, i.e. doing the actual plotting, and the body of this
    ## function is just to set up the observables.
    ##
    ## This will make it easier to manage each kind of graph I am showing. For example,
    ## suppose I want to show the cost function next to the population graph. Better to have
    ## that all in one function, rather than existing alongside the control plotting.
    #options = ["Graph 1", "Graph 2", "Graph 3"]
    #menu = Menu(fig[4,1], options=options)

    if !ismissing(prob)
        text!(infidelity_str_obsv, position=(0.0, 0.0), color=:red, font="Arial", fontsize=20)
    end

    return fig
end

"""
Get (approximately) the maximum control amplitude.

I am assuming that the control amplitude generally goes up as the control
parameters increase in value.

I am only using this function to help set the axis limits in the control
visualization, so it is fine if this is not very accurate.
"""
function get_max_control_amp(control, n_samples, pcof_upper_lim, pcof_lower_lim)
    pcof_upper = ones(get_number_of_control_parameters(control)) .* pcof_upper_lim
    pcof_lower = ones(get_number_of_control_parameters(control)) .* pcof_lower_lim

    ts = LinRange(0.0, control[1].tf, n_samples)

    p_vals_upper = [eval_p(control, t, pcof_upper) for t in ts]
    q_vals_upper = [eval_q(control, t, pcof_upper) for t in ts]

    p_vals_lower = [eval_p(control, t, pcof_lower) for t in ts]
    q_vals_lower = [eval_q(control, t, pcof_lower) for t in ts]

    control_val_lists = [p_vals_upper, q_vals_upper, p_vals_lower, q_vals_lower]
    max_control = maximum()

    
    max_q = maximum(p_vals)
end

end # Module
