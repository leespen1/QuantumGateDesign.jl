"""
    HermiteCarrierControl(N_points, tf, N_derivatives, carrier_wave_freqs; [scaling_type=:Heuristic])

Construct  a control that is the sum of Hermite interpolating polynomials of the values and first
`N_derivatives` derivatives at `N_points` evenly spaced points, multiplied by carrier waves. The control
vector gives the values and the derivatives (scaled depending on
`scaling_type`) for each of the polynomials multiplied by carrier waves.

# Notes
Working on making this non-allocating and non-repeating.

Also remember to eventually change when the 1/j! is applied, for better numerical
stability

Note: I can use a high-order hermite control for a low order method, and pcof still
works the same way.

And for pcof, it is convenient to just reshape a matrix whose columns are the derivatives

Static arrays could be useful here. Wouldn't have such a big struct, could just
construct them inline on the stack. Just need a N_derivatives struct parameter.

Idea: Have a version for which we specify N derivatives in pcof, but we use N+M
derivatives, which the remainder always being 0. That way we have highly smooth
controls, but we don't have as many control parameters. (right now, 3 carrier
waves, 2 points each, 3 derivatives each, uses 48 parameters.)

Also, as I have more derivatives actually controlled by pcof, the more
parameters I have affecting each time interval. Not sure how much that matters,
but with B-splines they seemed to think it was good that each point is affected
by at most 3 parameters. In my case, that number is 2*(1+N_derivatives)
"""
mutable struct HermiteCarrierControl <: AbstractControl
    N_coeff::Int64
    N_carriers::Int64
    N_coeffs_per_carrier::Int64
    tf::Float64
    N_points::Int64
    N_derivatives::Int64
    dt::Float64 # Should probably call this something else, to avoid confusion with timestep size
    Hmat::Matrix{Float64}
    fn_vals_p::Vector{Float64}
    fnp1_vals_p::Vector{Float64}
    fvals_collected_p::Vector{Float64}
    fn_vals_q::Vector{Float64}
    fnp1_vals_q::Vector{Float64}
    fvals_collected_q::Vector{Float64}
    uint_p::Vector{Float64}
    uint_q::Vector{Float64}
    ploc::Vector{Float64} # Just used for storage/working array. Values not important
    pcof_temp::Vector{Float64}
    coswt_derivatives::Vector{Float64}
    sinwt_derivatives::Vector{Float64}
    working_vec::Vector{Float64}
    carrier_wave_freqs::Vector{Float64}
    scaling_type::Symbol
    function HermiteCarrierControl(N_points::Int64, tf::Float64, N_derivatives::Int64, carrier_wave_freqs::AbstractVector{<: Real}, scaling_type::Symbol=:Heuristic)
        @assert N_points > 1

        N_coeffs_per_carrier = (N_derivatives+1)*N_points*2
        N_coeff = N_coeffs_per_carrier*length(carrier_wave_freqs)
        dt = tf / (N_points-1)
        Hmat = zeros(1+2*N_derivatives+1, 1+2*N_derivatives+1)
        # Hermite interpolant will be centered about the middle of each "control timestep"
        xl = 0.0
        xr = 1.0
        xc = 0.5
        icase = 0
        Hermite_map!(Hmat, N_derivatives, xl, xr, xc, icase)

        fn_vals_p = zeros(1+N_derivatives)
        fnp1_vals_p = zeros(1+N_derivatives)
        fvals_collected_p = zeros(2*N_derivatives + 2)

        fn_vals_q = zeros(1+N_derivatives)
        fnp1_vals_q = zeros(1+N_derivatives)
        fvals_collected_q = zeros(2*N_derivatives + 2)

        uint_p = zeros(2*N_derivatives + 2)
        uint_q = zeros(2*N_derivatives + 2)
        ploc = zeros(2*N_derivatives + 2) # Just used for storage/working array. Values not important
        
        pcof_temp = zeros(N_coeff)

        coswt_derivatives = zeros(1+N_derivatives)
        sinwt_derivatives = zeros(1+N_derivatives)
        control_derivatives = zeros(1+N_derivatives)
        working_vec = zeros(1+N_derivatives)

        carrier_wave_freqs = convert(Vector{Float64}, carrier_wave_freqs)
        N_carriers = length(carrier_wave_freqs)

        new(N_coeff, N_carriers, N_coeffs_per_carrier, 
            tf, N_points, N_derivatives, dt, Hmat, 
            fn_vals_p, fnp1_vals_p, fvals_collected_p, 
            fn_vals_q, fnp1_vals_q, fvals_collected_q, 
            uint_p, uint_q, ploc, pcof_temp,
            coswt_derivatives, sinwt_derivatives, 
            working_vec, carrier_wave_freqs, scaling_type)
    end
end



"""
Evaluate for a single carrier wave frequency
"""
function eval_derivative(control::HermiteCarrierControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, p_or_q::Symbol,
        w::Real
    )

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

    offset_n_p = 1 + (i-1)*(N_derivatives+1)
    offset_np1_p = 1 + i*(N_derivatives+1)
    offset_n_q = offset_n_p + div(control.N_coeffs_per_carrier, 2)
    offset_np1_q = offset_np1_p + div(control.N_coeffs_per_carrier, 2) 

    copyto!(control.fn_vals_p,   1, pcof, offset_n_p,   1+N_derivatives)
    copyto!(control.fnp1_vals_p, 1, pcof, offset_np1_p, 1+N_derivatives)
    copyto!(control.fn_vals_q,   1, pcof, offset_n_q,   1+N_derivatives)
    copyto!(control.fnp1_vals_q, 1, pcof, offset_np1_q, 1+N_derivatives)

    # Try to get scaling so that the parameters corresponding to higher
    # order derivatives have the same impact as the parameters
    # corresponding to lower order derivatives.
    for i in 0:control.N_derivatives
        if (control.scaling_type == :Taylor) # pcof elements are the taylor coefficients
            scaling_factor = 1
        elseif (control.scaling_type == :Derivative) # pcof elements are the derivative values
            scaling_factor = dt^i / factorial(i)
        elseif (control.scaling_type == :Heuristic) # pcof elements chosen based on experimentation
            scaling_factor = factorial(i+1)*2^i
            #scaling_factor = factorial(i)
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

    mul!(control.uint_p, control.Hmat, control.fvals_collected_p)
    mul!(control.uint_q, control.Hmat, control.fvals_collected_q)

    t_normalized = (t - t_center)/dt

    extrapolate!(control.uint_p, t_normalized, 2*m+1, control.ploc)
    extrapolate!(control.uint_q, t_normalized, 2*m+1, control.ploc)

    fill_coswt_derivatives!(control.coswt_derivatives, w, t, N_derivatives)
    fill_sinwt_derivatives!(control.sinwt_derivatives, w, t, N_derivatives)
    # Put in dt^i factors
    for i in 0:control.N_derivatives
        control.coswt_derivatives[1+i] *= dt^i
        control.sinwt_derivatives[1+i] *= dt^i
    end

    ret_val = 0.0
    if p_or_q == :p
        product_rule!(control.uint_p, control.coswt_derivatives, control.working_vec, N_derivatives)
        ret_val += control.working_vec[1+order] / (dt^order)
        product_rule!(control.uint_q, control.sinwt_derivatives, control.working_vec, N_derivatives)
        ret_val -= control.working_vec[1+order] / (dt^order)
    elseif p_or_q == :q
        product_rule!(control.uint_p, control.sinwt_derivatives, control.working_vec, N_derivatives)
        ret_val += control.working_vec[1+order] / (dt^order)
        product_rule!(control.uint_q, control.coswt_derivatives, control.working_vec, N_derivatives)
        ret_val += control.working_vec[1+order] / (dt^order)
    else
        throw(DomainError(p_or_q, "Must be :p or :q"))
    end

    return ret_val
end



"""
Given vectors of derivatives for x and y (in the usual 1/j! format), compute
the derivatives of x*y.

I think you should be able to have dt^j factors and the output will be in the
same form (x'''y' -> dt^4x'''y').  That simplifies things.
"""
function product_rule!(x_derivatives, y_derivatives, prod_derivatives, N_derivatives=missing)
    if ismissing(N_derivatives)
        N_derivatives = length(prod_derivatives)-1
    end

    prod_derivatives[1:1+N_derivatives] .= 0
    # If there are extra entries in the results vector, set them to NaN
    prod_derivatives[1+N_derivatives+1:end] .= NaN 

    for dn in 0:N_derivatives # dn is the derivative order we want to compute
        for k in 0:dn
            prod_derivatives[1+dn] += x_derivatives[1+k]*y_derivatives[1+dn-k]
        end
    end
    
    return prod_derivatives
end



function fill_coswt_derivatives!(coswt_derivatives, w, t, N_derivatives)
    coswt_derivatives .= NaN # Set everything to NaN, so we can see if we don't fill whole array
    for derivative_order in 0:N_derivatives
        if iseven(derivative_order)
            coswt_derivatives[1+derivative_order] = cos(w*t)
        else
            coswt_derivatives[1+derivative_order] = sin(w*t)
        end
        coswt_derivatives[1+derivative_order] *= w^derivative_order
        if (derivative_order % 4) in (1,2)
            coswt_derivatives[1+derivative_order] *= -1
        end
        coswt_derivatives[1+derivative_order] /= factorial(derivative_order)
    end
    return coswt_derivatives
end



"""
Compute derivatives of sin(ωt)
"""
function fill_sinwt_derivatives!(sinwt_derivatives, w, t, N_derivatives)
    sinwt_derivatives .= NaN # Set everything to NaN, so we can see if we don't fill whole array
    for derivative_order in 0:N_derivatives
        if iseven(derivative_order)
            sinwt_derivatives[1+derivative_order] = sin(w*t)
        else
            sinwt_derivatives[1+derivative_order] = cos(w*t)
        end
        sinwt_derivatives[1+derivative_order] *= w^derivative_order
        if (derivative_order % 4) in (2,3)
            sinwt_derivatives[1+derivative_order] *= -1
        end
        sinwt_derivatives[1+derivative_order] /= factorial(derivative_order)
    end
    return sinwt_derivatives
end


#=
"""
Takes array pcof input. So can reshape nicely. 
"""
function eval_derivative(control::HermiteCarrierControl, t::Real,
        pcof::AbstractArray{<: Real},  order::Int64, p_or_q::Symbol,
        w::Real=0.0
    )
    # Reshape matrix or higher dimension array to vector 
    # (only does 2 allocs, 80 bytes, independent of size)
    pcof_vec = reshape(pcof, :) 
    return eval_derivative(control, t, pcof_vec, order, p_or_q, w)
end
=#

function eval_p(control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    derivative_order = 0
    return eval_p_derivative(control, t, pcof, derivative_order)
end

function eval_q(control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    derivative_order = 0
    return eval_q_derivative(control, t, pcof, derivative_order)
end

function eval_p_derivative(control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    ret_val = 0.0
    for carrier_index in 1:control.N_carriers
        w = control.carrier_wave_freqs[carrier_index]
        carrier_offset = (carrier_index-1)*control.N_coeffs_per_carrier
        this_carrier_pcof = view(pcof, 1+carrier_offset:carrier_offset+control.N_coeffs_per_carrier)
        ret_val += eval_derivative(control, t, this_carrier_pcof, order, :p, w)
    end
    return ret_val
end

function eval_q_derivative(control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    ret_val = 0.0
    for carrier_index in 1:control.N_carriers
        w = control.carrier_wave_freqs[carrier_index]
        carrier_offset = (carrier_index-1)*control.N_coeffs_per_carrier
        this_carrier_pcof = view(pcof, 1+carrier_offset:carrier_offset+control.N_coeffs_per_carrier)
        ret_val += eval_derivative(control, t, this_carrier_pcof, order, :q, w)
    end
    return ret_val
end

"""
Should actually be very easy to get just the partial derivative w.r.t a single
coefficient for this kind of control.

I could make this more efficient by only doing this for the pcof entries which
affect p/q and this value of t

KEEP IN MIND THAT P AND Q 'MIX' NOW
"""
function eval_grad_p_derivative!(grad::AbstractVector{Float64},
        control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64
    )

    grad .= 0

    i = find_region_index(t, control.tf, control.N_points-1)
    region_offset = 1+(i-1)*(control.N_derivatives+1)
    q_offset = div(control.N_coeffs_per_carrier, 2)

    # Don't need entire pcof, since we only evaluate one carrier at a time
    pcof_temp_view  = view(control.pcof_temp, 1:control.N_coeffs_per_carrier)

    for carrier_index in 1:control.N_carriers
        carrier_offset = (carrier_index-1)*control.N_coeffs_per_carrier
        w = control.carrier_wave_freqs[carrier_index]
        total_offset = region_offset + carrier_offset

        control.pcof_temp .= 0

        for k in 0:1+2*control.N_derivatives
            pcof_temp_view .= 0
            pcof_temp_view[region_offset+k] = 1
            grad[total_offset+k] = eval_derivative(control, t, pcof_temp_view, order, :p, w)
            pcof_temp_view .= 0
            pcof_temp_view[region_offset+q_offset+k] = 1
            grad[total_offset+q_offset+k] = eval_derivative(control, t, pcof_temp_view, order, :p, w)
        end
    end

    return grad
end



function eval_grad_q_derivative!(grad::AbstractVector{Float64},
        control::HermiteCarrierControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64
    )

    grad .= 0

    i = find_region_index(t, control.tf, control.N_points-1)
    region_offset = 1+(i-1)*(control.N_derivatives+1)
    q_offset = div(control.N_coeffs_per_carrier, 2)

    # Don't need entire pcof, since we only evaluate one carrier at a time
    pcof_temp_view  = view(control.pcof_temp, 1:control.N_coeffs_per_carrier)

    for carrier_index in 1:control.N_carriers
        carrier_offset = (carrier_index-1)*control.N_coeffs_per_carrier
        w = control.carrier_wave_freqs[carrier_index]
        total_offset = region_offset + carrier_offset

        control.pcof_temp .= 0

        for k in 0:1+2*control.N_derivatives
            pcof_temp_view .= 0
            pcof_temp_view[region_offset+k] = 1
            grad[total_offset+k] = eval_derivative(control, t, pcof_temp_view, order, :q, w)
            pcof_temp_view .= 0
            pcof_temp_view[region_offset+q_offset+k] = 1
            grad[total_offset+q_offset+k] = eval_derivative(control, t, pcof_temp_view, order, :q, w)
        end
    end

    return grad
end
