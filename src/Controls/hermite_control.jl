"""
Working on making this non-allocating and non-repeating.

Also remember to eventually change when the 1/j! is applied, for better numerical
stability

Note: I can use a high-order hermite control for a low order method, and pcof still
works the same way.

And for pcof, it is convenient to just reshape a matrix whose columns are the derivatives
"""
mutable struct HermiteControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_points::Int64
    N_derivatives::Int64
    dt::Float64
    Hmat::Matrix{Float64}
    tn::Float64
    tnp1::Float64
    t_current::Float64
    fn_vals_p::Vector{Float64}
    fnp1_vals_p::Vector{Float64}
    fvals_collected_p::Vector{Float64}
    fn_vals_q::Vector{Float64}
    fnp1_vals_q::Vector{Float64}
    fvals_collected_q::Vector{Float64}
    uint_p::Vector{Float64}
    uint_q::Vector{Float64}
    uint_intermediate_p::Vector{Float64}
    uint_intermediate_q::Vector{Float64}
    ploc::Vector{Float64} # Just used for storage/working array. Values not important
    function HermiteControl(N_points::Int64, tf::Float64, N_derivatives::Int64)
        @assert N_points > 1

        N_coeff = N_points*(N_derivatives+1)*2
        dt = tf / (N_points-1)
        Hmat = zeros(1+2*N_derivatives+1, 1+2*N_derivatives+1)
        # Hermite interpolant will be centered about the middle of each "control timestep"
        xl = 0.0
        xr = 1.0
        xc = 0.5
        icase = 0
        Hermite_map!(Hmat, N_derivatives, xl, xr, xc, icase)

        # Dummy time vals
        tn = NaN
        tnp1 = NaN
        t_current = NaN

        fn_vals_p = zeros(1+N_derivatives)
        fnp1_vals_p = zeros(1+N_derivatives)
        fvals_collected_p = zeros(2*N_derivatives + 2)

        fn_vals_q = zeros(1+N_derivatives)
        fnp1_vals_q = zeros(1+N_derivatives)
        fvals_collected_q = zeros(2*N_derivatives + 2)

        uint_p = zeros(2*N_derivatives + 2)
        uint_q = zeros(2*N_derivatives + 2)
        uint_intermediate_p = zeros(2*N_derivatives + 2)
        uint_intermediate_q = zeros(2*N_derivatives + 2)
        ploc = zeros(2*N_derivatives + 2) # Just used for storage/working array. Values not important


        new(N_coeff, tf, N_points, N_derivatives, dt, Hmat, tn, tnp1, t_current,
            fn_vals_p, fnp1_vals_p, fvals_collected_p, 
            fn_vals_q, fnp1_vals_q, fvals_collected_q, 
            uint_p, uint_q, uint_intermediate_p, uint_intermediate_q, ploc)
    end
end

function eval_p(control::HermiteControl, t::Real, pcof::AbstractVector{<: Real})
    derivative_order = 0
    return eval_p_derivative(control, t, pcof, derivative_order)
end

function eval_q(control::HermiteControl, t::Real, pcof::AbstractVector{<: Real})
    derivative_order = 0
    return eval_q_derivative(control, t, pcof, derivative_order)
end

function eval_p_derivative(control::HermiteControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_derivative(control, t, pcof, order, :p)
end

function eval_q_derivative(control::HermiteControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    return eval_derivative(control, t, pcof, order, :q)
end


"""
New, non-allocating version. Currently bugged (not giving pcof[1] at t=0.0)
"""
function eval_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, p_or_q)

    # Maybe remove this warning for speed.
    if (order > (2+2*control.N_derivatives)-1)
        throw(DomainError(order, "Derivative order must not exceed 2*(1+N_derivatives)"))
    end

    k = order
    m = control.N_derivatives
    N_derivatives = control.N_derivatives

    i = find_region_index(t, control.tf, control.N_points-1)

    dt = control.dt

    tn = dt*(i-1)
    t_center = tn + 0.5*dt
    tnp1 = dt*i

    # Check if we need to update the interval
    if (control.tn != tn || control.tnp1 != tnp1)
        
        control.tn = tn
        control.tnp1 = tnp1

        offset_n_p = 1 + (i-1)*(N_derivatives+1)
        offset_np1_p = 1 + i*(N_derivatives+1)
        offset_n_q = offset_n_p + div(control.N_coeff, 2)
        offset_np1_q = offset_np1_p + div(control.N_coeff, 2)

        copyto!(control.fn_vals_p,   1, pcof, offset_n_p,   1+N_derivatives)
        copyto!(control.fnp1_vals_p, 1, pcof, offset_np1_p, 1+N_derivatives)
        copyto!(control.fn_vals_q,   1, pcof, offset_n_q,   1+N_derivatives)
        copyto!(control.fnp1_vals_q, 1, pcof, offset_np1_q, 1+N_derivatives)

        for i in 0:control.N_derivatives
            control.fn_vals_p[1+i] *= dt^i / factorial(i)
            control.fnp1_vals_p[1+i] *= dt^i / factorial(i)
            control.fn_vals_q[1+i] *= dt^i / factorial(i)
            control.fnp1_vals_q[1+i] *= dt^i / factorial(i)
        end

        copyto!(control.fvals_collected_p, 1, control.fn_vals_p, 1, 1+N_derivatives)
        copyto!(control.fvals_collected_p, 2+N_derivatives, control.fnp1_vals_p, 1, 1+N_derivatives)
        copyto!(control.fvals_collected_q, 1, control.fn_vals_q, 1, 1+N_derivatives)
        copyto!(control.fvals_collected_q, 2+N_derivatives, control.fnp1_vals_q, 1, 1+N_derivatives)

        mul!(control.uint_intermediate_p, control.Hmat, control.fvals_collected_p)
        mul!(control.uint_intermediate_q, control.Hmat, control.fvals_collected_q)
    end

    # Check if we need to update the time within the interval 
    # This allows us to reuse computation when computing different derivatives at same time
    if (t != control.t_current)
        control.t_current = t

        t_normalized = (t - t_center)/dt

        control.uint_p .= control.uint_intermediate_p
        control.uint_q .= control.uint_intermediate_q
        extrapolate!(control.uint_p, t_normalized, 2*m+1, control.ploc)
        extrapolate!(control.uint_q, t_normalized, 2*m+1, control.ploc)
    end

    
    # Decide which function (p or q) and what derivative to return
    ret_val = NaN
    if p_or_q == :p
        ret_val = control.uint_p[1+order] * factorial(order) / (dt^(order)) # Want the function value, although the dt^order/order! would be useful in the remaining computation
    elseif p_or_q == :q
        ret_val =  control.uint_q[1+order] * factorial(order) / (dt^(order)) # Want the function value, although the dt^order/order! would be useful in the remaining computation
    else
        throw(DomainError(p_or_q, "Must be :p or :q"))
    end

    return ret_val
end


"""
For making a hermite interpolation control based on an existing control.
"""
function construct_pcof_from_sample(control_orig, pcof_orig, hermite_control)
    N_derivatives = hermite_control.N_derivatives
    N_samples = div(hermite_control.N_coeff, 2*(N_derivatives+1))

    pcof_mat = zeros(1+N_derivatives, N_samples, 2)
    for sample_index in 1:N_samples
        t = (sample_index-1)*hermite_control.dt
        for derivative_order in 0:N_derivatives
            pcof_mat[1+derivative_order, sample_index, 1] = eval_p_derivative(control_orig, t, pcof_orig, derivative_order)
            pcof_mat[1+derivative_order, sample_index, 2] = eval_q_derivative(control_orig, t, pcof_orig, derivative_order)
        end
    end

    pcof_vec = reshape(pcof_mat, length(pcof_mat))
    return pcof_vec
end
