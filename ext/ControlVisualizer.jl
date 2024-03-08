module ControlVisualizer

println("Loading ControlVisualizer")

using QuantumGateDesign
using GLMakie


"""
Visualize a Control Using Makie

This is really old. From back when I was using functions as parameters for the
object instead of methods for the type. Needs to be updated to work.
"""
function QuantumGateDesign.visualize_control(control; n_points=101, prob=missing, 
        pcof_init=missing, use_tboxes=true)

    tf = control.tf

    t_range_control = LinRange(0, tf, n_points)

    # Set up Makie figure and axis for plotting the control
    fig = Figure(;)
    ax = Axis(fig[1,1], xlabel="Time (ns)", ylabel="Control Amplitude (Unitless)")
    xlims!(ax, -(1/16)*tf, tf*(1+(1/16))) # Put a little bit of padding outside the control range

    #==================================
    # Set up control vector slidergrid
    ==================================#

    # Set up control vector pcof, and slidergrid for manipulating control vector
    pcof_obsv = Vector{Observable{Float64}}(undef, control.N_coeff)

    # Set initial values for control vector. Initialize to user-specified values, if given
    startvalues = zeros(control.N_coeff)
    if !ismissing(pcof_init)
        startvalues .= pcof_init
    end

    #pcof_range = LinRange(0, 1e-3, 11)
    pcof_range = LinRange(0, 1, 101)

    pcof_slider_parameters = [
        (label="pcof[$i]", range=pcof_range, startvalue=startvalues[i])
        for i in 1:control.N_coeff
    ]

    pcof_slidergrid = SliderGrid(fig[2,1], pcof_slider_parameters...)

    # Set up link between slider values and pcof observable values
    for i in 1:control.N_coeff
        pcof_obsv[i] = lift(x -> x, pcof_slidergrid.sliders[i].value)
    end

    # Set up function value observables
    p_vals_obsv = Observable([Point2(t_range_control[k], eval_p(control, t_range_control[k], startvalues)*1e3) for k in 1:n_points])
    q_vals_obsv = Observable([Point2(t_range_control[k], eval_q(control, t_range_control[k], startvalues)*1e3) for k in 1:n_points])

    # If prob is provided, also plot population evolution
    if !ismissing(prob)
        @assert tf == prob.tf # Make sure problem and control have same range
        t_range_prob = LinRange(0, tf, 1+prob.nsteps)

        # Set up plot for populations
        ax_population = Axis(fig[3,1], xlabel="Time (ns)", ylabel="Populations")
        xlims!(ax_population, -(1/16)*tf, tf*(1+(1/16)))

        # Set up population value observables
        # Each initial condition has it's own population array
        populations_obsv_list = [Observable(zeros(size(prob.u0, 1), 1+prob.nsteps)) for i in 1:size(prob.u0, 2)]

    end

    #=================================================
    # Function for updating graph if parameters change
    =================================================#
    function update_graph()
        pcof = to_value.(pcof_obsv)
        # (in units of MHz, non angular)
        p_vals_obsv[] = [Point2(t_range_control[k], eval_p(control, t_range_control[k], pcof)) for k in 1:n_points]
        q_vals_obsv[] = [Point2(t_range_control[k], eval_q(control, t_range_control[k], pcof)) for k in 1:n_points]

        max_p = maximum(getindex.(p_vals_obsv[], 2))
        max_q = maximum(getindex.(q_vals_obsv[], 2))
        min_p = minimum(getindex.(p_vals_obsv[], 2))
        min_q = minimum(getindex.(q_vals_obsv[], 2))

        # Update ylims to capture entire control; const args ensure ylims != (0,0)
        ylims!(ax, min(min_p, min_q)-1, max(max_p, max_q)+1)

        if !ismissing(prob)
            history = eval_forward(prob, control, pcof, order=4)
            populations = get_populations(history)
            for i in 1:size(prob.u0, 2)
                populations_obsv_list[i][] = populations[:,:,i]
            end
        end
    end

    # Initial update
    update_graph()


    # Set up link so that graph updates when any control parameter changes
    for i in 1:control.N_coeff
        on(pcof_obsv[i]) do coeff
            update_graph()
        end
    end

    # Draw the control functions
    GLMakie.lines!(ax, p_vals_obsv, linewidth=2)
    GLMakie.lines!(ax, q_vals_obsv, linewidth=2)

    # Draw the populations
    if !ismissing(prob)
        #for i in 1:size(prob.u0, 2)
        for i in 1:1
            GLMakie.series!(ax_population, t_range_prob, populations_obsv_list[i])
        end
    end


    # For changing layouts. Ideally, I should have functions for constructing
    # the various layouts, i.e. doing the actual plotting, and the body of this
    # function is just to set up the observables.
    #
    # This will make it easier to manage each kind of graph I am showing. For example,
    # suppose I want to show the cost function next to the population graph. Better to have
    # that all in one function, rather than existing alongside the control plotting.
    options = ["Graph 1", "Graph 2", "Graph 3"]
    menu = Menu(fig[4,1], options=options)


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
    pcof_upper = ones(control.N_coeff) .* pcof_upper_lim
    pcof_lower = ones(control.N_coeff) .* pcof_lower_lim

    ts = LinRange(0.0, control.tf, n_samples)

    p_vals_upper = [eval_p(control, t, pcof_upper) for t in ts]
    q_vals_upper = [eval_q(control, t, pcof_upper) for t in ts]

    p_vals_lower = [eval_p(control, t, pcof_lower) for t in ts]
    q_vals_lower = [eval_q(control, t, pcof_lower) for t in ts]

    control_val_lists = [p_vals_upper, q_vals_upper, p_vals_lower, q_vals_lower]
    max_control = maximum()

    
    max_q = maximum(p_vals)
end

end # Module
