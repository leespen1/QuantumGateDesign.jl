module ControlVisualizer

using HermiteOptimalControl
using GLMakie
#using CairoMakie


function HermiteOptimalControl.visualize_control(control, tf, npoints; prob=missing, 
        pcof_init=missing, use_slidergrid=true, use_tboxes=true)
    # Set up Makie figure and axis for plotting
    fig = Figure(;)
    ax = Axis(fig[1,1], xlabel="Time (ns)", ylabel="Control Amplitude (MHz)")

    # Set up control vector pcof, and slidergrid for manipulating control vector
    pcof_o = Vector{Observable{Float64}}(undef, control.N_coeff)

    startvalues = zeros(control.N_coeff)
    if !ismissing(pcof_init)
        startvalues .= pcof_init
    end

    # Set up timing observables
    n_points_o = Observable{Int64}(npoints)
    tf_o = Observable{Float64}(tf)

    # Time grid tracks final time
    # Having the number of points in linrange seems to produce errors, but the plot still works
    ts_o = lift((t,n) -> LinRange(0,t,n), tf_o, n_points_o)

    pcof_range = LinRange(0, 1e-3, 10)

    if use_slidergrid
        pcof_slider_parameters = [
            (label="pcof[$i]", range=pcof_range, startvalue=startvalues[i])
            for i in 1:control.N_coeff
        ]
        pcof_slidergrid = SliderGrid(fig[4,1], pcof_slider_parameters...)

        # Set up link between slider values and pcof observable values
        for i in 1:control.N_coeff
            pcof_o[i] = lift(x -> x, pcof_slidergrid.sliders[i].value)
        end
    elseif !ismissing(pcof_init)
        for i in 1:control.N_coeff
            pcof_o[i] = Observable{Float64}(pcof_init[i])
        end
    else
        throw("use_slidergrid=false, but no initial control vector provided")
    end



    if use_tboxes
        # Setup textbox for modification of final time, and number of grid points
        tb_tf = Textbox(fig[2,1], placeholder= "Enter final time", validator=Float64, tellwidth=false)
        on(tb_tf.stored_string) do s
            tf_o[] = parse(Float64, s)
        end

        tb_npoints = Textbox(fig[3,1], placeholder= "Number of gridpoints", validator=Int64, tellwidth=false)
        on(tb_npoints.stored_string) do s
            n_points_o[] = parse(Int64, s)
        end
    end

    # Set up function value observables
    p_vals_o = Observable(zeros(n_points_o[]))
    q_vals_o = Observable(zeros(n_points_o[]))

    # Set up function for updating values of p and q when coefficients or times change
    # (in units of MHz, non angular)
    function update_graph()
        pcof = to_value.(pcof_o)
        p_vals_o[] = [control.p[1](ts_o[][k], pcof)*1e3 for k in 1:n_points_o[]]
        q_vals_o[] = [control.q[1](ts_o[][k], pcof)*1e3 for k in 1:n_points_o[]]

        # Update ylims to capture entire control; const args ensure ylims != (0,0)
        ylims!(ax, min(minimum(p_vals_o[]), minimum(q_vals_o[]))-1,
                   max(maximum(p_vals_o[]), maximum(q_vals_o[]))+1)

        #=
        if !ismissing(prob)
            history = eval_forward(prob, control, pcof, order=4)
            populations = get_populations(history)
            for i in 1:prob.N_ess_levels
                populations_o[i][] = populations[i,:]
            end
        end
        =#
    end

    # Initial update
    update_graph()

    # Set xlims to track final time (and leave a fraction of the window empty)
    on(tf_o) do tf
        xlims!(ax, -(1/16)*tf, tf*(1+(1/16)))
        update_graph()
    end

    on(n_points_o) do n_points
        update_graph()
    end

    # Set up population value observables
    #=
    if !ismissing(prob)
        populations_o = [Observable(zeros(n_points_o[])) for i in 1:prob.N_ess_levels]
    end
    =#


    # set p/q_vals to change if any of the coefficients change
    for i in 1:control.N_coeff
        on(pcof_o[i]) do coeff
            update_graph()
        end
    end


    # Draw the functions
    Makie.lines!(ax, ts_o, p_vals_o, linewidth=2)
    Makie.lines!(ax, ts_o, q_vals_o, linewidth=2)
    #=
    if !ismissing(prob)
        for i in 1:prob.
            Makie.lines!(ax, ts_o, populations_o[i])
        end
    end
    =#

    return fig
end


end # Module
