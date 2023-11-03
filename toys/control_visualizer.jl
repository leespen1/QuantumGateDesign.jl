using GLMakie
using HermiteOptimalControl

function visualize_control(control, range=LinRange(0,1,11))
    # Set up Makie figure and axis for plotting
    fig = Figure(; resolution=(800,800))
    ax = Axis(fig[1,1])

    # Set up control vector pcof, and slidergrid for manipulating control vector
    pcof_o = Vector{Observable{Float64}}(undef, control.N_coeff)
    pcof_slider_parameters = [
        (label="pcof[$i]", range=range, startvalue=0.0)
        for i in 1:control.N_coeff
    ]
    pcof_slidergrid = SliderGrid(fig[2,1], pcof_slider_parameters...)

    # Set up link between slider values and pcof observable values
    for i in 1:control.N_coeff
        pcof_o[i] = lift(x -> x, pcof_slidergrid.sliders[i].value)
    end


    # Set up timing observables
    n_points_o = Observable{Int64}(101)
    tf_o = Observable{Float64}(2*pi)
    ts_o = lift((t,n) -> LinRange(0,t,n), tf_o, n_points_o)

    # Set xlims to track final time (and leave a fraction of the window empty)
    on(tf_o) do tf
        xlims!(ax, -(1/16)*tf, tf*(1+(1/16)))
    end

    # Setup textbox for modification of final time
    tb = Textbox(fig[3,1], placeholder= "Entre final time", validator=Float64, tellwidth=false)
    on(tb.stored_string) do s
        tf_o[] = parse(Float64, s)
    end

    # Set up function value observables
    p_vals_o = Observable(zeros(n_points_o[]))
    q_vals_o = Observable(zeros(n_points_o[]))

    # set p/q_vals to change if any of the coefficients change
    for i in 1:control.N_coeff
        on(pcof_o[i]) do coeff
            pcof = to_value.(pcof_o)
            p_vals_o[] = [control.p[1](ts_o[][k], pcof) for k in 1:n_points_o[]]
            q_vals_o[] = [control.q[1](ts_o[][k], pcof) for k in 1:n_points_o[]]

            # Update ylims to capture entire control; const args ensure ylims != (0,0)
            ylims!(ax, min(minimum(p_vals_o[]), minimum(q_vals_o[]))-1,
                       max(maximum(p_vals_o[]), maximum(q_vals_o[]))+1)
        end
    end


    # Draw the functions
    Makie.lines!(ax, ts_o, p_vals_o)
    Makie.lines!(ax, ts_o, q_vals_o)

    return fig
end


