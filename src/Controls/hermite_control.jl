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
    pcof_temp::Vector{Float64}
    scaling_type::Symbol
    function HermiteControl(N_points::Int64, tf::Float64, N_derivatives::Int64, scaling_type::Symbol=:Heuristic)
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
        
        pcof_temp = zeros(N_coeff)


        new(N_coeff, tf, N_points, N_derivatives, dt, Hmat, tn, tnp1, t_current,
            fn_vals_p, fnp1_vals_p, fvals_collected_p, 
            fn_vals_q, fnp1_vals_q, fvals_collected_q, 
            uint_p, uint_q, uint_intermediate_p, uint_intermediate_q, ploc,
            pcof_temp, scaling_type)
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
Should actually be very easy to get just the partial derivative w.r.t a single
coefficient for this kind of control.

I could make this more efficient by only doing this for the pcof entries which
affect p/q and this value of t
"""
function eval_grad_p_derivative!(grad::AbstractVector{Float64},
        control::HermiteControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64
    )

    grad .= 0

    i = find_region_index(t, control.tf, control.N_points-1)

    offset = 1 + (i-1)*(control.N_derivatives+1)

    # Get effect of control points and derivatives for both ends of the interval
    for k in 0:1+2*control.N_derivatives
        control.pcof_temp .= 0
        control.pcof_temp[offset+k] = 1
        grad[offset+k] = eval_p_derivative(control, t, control.pcof_temp, order)
    end

    return grad
end

function eval_grad_q_derivative!(grad::AbstractVector{Float64},
        control::HermiteControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)

    grad .= 0

    i = find_region_index(t, control.tf, control.N_points-1)

    offset = 1 + (i-1)*(control.N_derivatives+1) + div(control.N_coeff, 2)

    # Get effect of control points and derivatives for both ends of the interval
    for k in 0:1+2*control.N_derivatives
        control.pcof_temp .= 0
        control.pcof_temp[offset+k] = 1
        grad[offset+k] = eval_q_derivative(control, t, control.pcof_temp, order)
    end

    return grad
end

function eval_grad_p_derivative( control::HermiteControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    grad = Vector{Float64}(undef, length(pcof))
    eval_grad_p_derivative!(grad, control, t, pcof, order)
    return grad
end

function eval_grad_q_derivative( control::HermiteControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64)

    grad = Vector{Float64}(undef, length(pcof))
    eval_grad_q_derivative!(grad, control, t, pcof, order)
    return grad
end

"""
Should actually be very easy to get just the partial derivative w.r.t a single
coefficient for this kind of control.

I could make this more efficient by only doing this for the pcof entries which
affect p/q and this value of t
"""
function eval_grad_p_derivative(control::HermiteControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64, pcof_index::Int)
    control.pcof_temp .= 0
    control.pcof_temp[pcof_index] = 1
    return eval_p_derivative(control, t, control.pcof_temp, order)
end

function eval_grad_q_derivative(control::HermiteControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64, pcof_index::Int)

    control.pcof_temp .= 0
    control.pcof_temp[pcof_index] = 1
    return eval_q_derivative(control, t, control.pcof_temp, order)
end


#TODO huge risk of incorrect results when using a vector of these controls. Will
#not update fn_vals/fnp1_vals when calling the same control for different pcofs
"""
New, non-allocating version. Currently bugged (not giving pcof[1] at t=0.0)
"""
function eval_derivative(control::HermiteControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, p_or_q::Symbol;
        force_refresh::Bool=true
    )

    # Maybe remove this warning for speed.
    if (order >= 2*(1+control.N_derivatives))
        return 0.0
        #throw(DomainError(order, "Derivative order must not exceed 2*(1+N_derivatives)"))
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
    if (control.tn != tn || control.tnp1 != tnp1 || force_refresh)
        
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

        # Try to get scaling so that the parameters corresponding to higher
        # order derivatives have the same impact as the parameters
        # corresponding to lower order derivatives.
        for i in 0:control.N_derivatives
            if (control.scaling_type == :Taylor)
                scaling_factor = 1
            elseif (control.scaling_type == :Derivative)
                scaling_factor = dt^i / factorial(i)
            elseif (control.scaling_type == :Heuristic)
                scaling_factor = factorial(i+1)*2^i
            else
                throw(ArgumentError(string(control.scaling_type)))
            end

            control.fn_vals_p[1+i]   *= scaling_factor
            control.fnp1_vals_p[1+i] *= scaling_factor
            control.fn_vals_q[1+i]   *= scaling_factor
            control.fnp1_vals_q[1+i] *= scaling_factor
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
    if (t != control.t_current || force_refresh)
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

"""
Same as above, but for multiple controls, and creates hermite controls
"""
function sample_from_controls(controls_orig, pcof_orig, N_samples, N_derivatives)
    controls_new = Vector{HermiteControl}()
    full_pcof_new = Vector{Float64}()
    for (i, control_orig) in enumerate(controls_orig)
        # Create control
        local_hermite_control = HermiteControl(N_samples, control_orig.tf, N_derivatives) 
        push!(controls_new, local_hermite_control)

        # Sample original control for values and derivatives at sample points
        local_pcof_orig = get_control_vector_slice(pcof_orig, controls_orig, i)
        local_pcof_new = construct_pcof_from_sample(control_orig, local_pcof_orig, local_hermite_control)

        # Append local control vector to global control vector
        full_pcof_new = vcat(full_pcof_new, local_pcof_new)
    end

    return controls_new, full_pcof_new
end
