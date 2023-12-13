struct HermiteControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_points::Int64
    N_derivatives::Int64
    dt::Float64
    Hmat::Matrix{Float64}
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
        new(N_coeff, tf, N_points, N_derivatives, dt, Hmat)
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



function eval_derivative(control::AbstractControl, t::Real,
        pcof::AbstractVector{<: Real},  order::Int64, p_or_q)
    if (order > control.N_derivatives)
        #return ForwardDiff.derivative(t_dummy -> eval_derivative(control, t_dummy, pcof, order-1, p_or_q), t)
        throw(DomainError(order))
    end

    k = order
    m = control.N_derivatives

    i = find_region_index(t, control.tf, control.N_points-1)

    dt = control.dt

    tn = dt*(i-1)
    t_center = tn + 0.5*dt
    tnp1 = dt*i
    #println("t=$t, i=$i, tn=$tn, tnp1=$tnp1")

    offset_n = 1 + (i-1)*(control.N_derivatives+1)
    offset_np1 = 1 + i*(control.N_derivatives+1)

    # Use second half of control vector for q
    if (p_or_q == :q)
        offset_n   += div(control.N_coeff, 2)
        offset_np1 += div(control.N_coeff, 2)
    elseif (p_or_q != :p)
        throw(DomainError(p_or_q))
    end

    fn_vals = copy(pcof[offset_n : offset_n + control.N_derivatives])
    fnp1_vals = copy(pcof[offset_np1 : offset_np1 + control.N_derivatives])

    for i in 0:control.N_derivatives
        fn_vals[1+i] *= dt^i / factorial(i)
        fnp1_vals[1+i] *= dt^i / factorial(i)
    end


    uint = zeros(2*m + 2)
    fvals_collected = vcat(fn_vals, fnp1_vals)
    mul!(uint, control.Hmat, fvals_collected)

    # For more efficiency, could store uint and reuse it if the next function
    # evaluation is in the same control interval.
    # Then only extrapolate would need to be called.

    t_normalized = (t - t_center)/dt

    ploc = zeros(2*m+2) # Just used for storage/working array. Values not important
    extrapolate!(uint, t_normalized, 2*m+1, ploc)


    return uint[1+order] * factorial(order) / (dt^(order)) # Want the function value, although the dt^order/order! would be useful in the remaining computation
end
