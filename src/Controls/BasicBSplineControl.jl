"""
The first version will assume no repeating knots. You just give the number of
knots / segments.
"""
struct BasicBSplineControl{degree} <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_knots::Int64
    P::BasicBSpline.BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}
    P1::BSplineDerivativeSpace{1, BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}, Float64}
    P2::BSplineDerivativeSpace{2, BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}, Float64}
    P3::BSplineDerivativeSpace{3, BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}, Float64}
    P4::BSplineDerivativeSpace{4, BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}, Float64}
    function BasicBSplineControl(degree::Integer, tf::Real, N_knots::Integer)
        degree = convert(Int64, degree)
        tf = convert(Float64, tf)
        N_knots = convert(Int64, N_knots)
        N_coeff = 2*N_knots
        # Make the knots on the interval [0,1], then scale the inputs t when evaluating
        knot_vector = BasicBSpline.KnotVector(LinRange(0, 1, N_knots))
        P = BSplineSpace{degree}(knot_vector)
        P1 = BSplineDerivativeSpace{1}(P)
        P2 = BSplineDerivativeSpace{2}(P)
        P3 = BSplineDerivativeSpace{3}(P)
        P4 = BSplineDerivativeSpace{4}(P)
        new{degree}(N_coeff, tf, N_knots, P, P1, P2, P3, P4)
    end
end

function eval_p(control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real})
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(control.P, t_scaled)
    for (n, basis_val) in enumerate(bspline_basis_vals)
        # Multiply basis function value by control point value
        bspline_val += pcof[i-1+n] * basis_val
    end

    return bspline_val
end

function eval_q(control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real})
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_coeff, 2)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(control.P, t_scaled)
    for (n, basis_val) in enumerate(bspline_basis_vals)
        # Multiply basis function value by control point value
        bspline_val += pcof[offset+i-1+n] * basis_val
    end

    return bspline_val
end

function eval_p_derivative(
        control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real},
        order::Integer
    )
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)


    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[1+order], t_scaled)
    for (n, basis_val) in enumerate(bspline_basis_vals)
        # Multiply basis function value by control point value
        bspline_val += pcof[i-1+n] * basis_val
    end

    return bspline_val
end

function eval_q_derivative(
        control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real},
        order::Integer
    )
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)


    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_ceoff, 2)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[1+order], t_scaled)
    for (n, basis_val) in enumerate(bspline_basis_vals)
        # Multiply basis function value by control point value
        bspline_val += pcof[offset+i-1+n] * basis_val
    end

    return bspline_val
end


"""
It is assumed we will not compute derivatives higher than order 4 (i.e. don't use
a numerical method of order greater than 10)
"""
function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::BasicBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}
    ) 
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_coeff, 2)

    for k in eachindex(vals_vec)
        bspline_val = 0.0
        bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[k], t_scaled)

        for (n, basis_val) in enumerate(bspline_basis_vals)
            # Multiply basis function value by control point value
            bspline_val += pcof[i-1+n] * basis_val
        end

        vals_vec[k] = bspline_val
    end
    return vals_vec
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::BasicBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}
    ) 
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_coeff, 2)

    for k in eachindex(vals_vec)
        bspline_val = 0.0
        bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[k], t_scaled)

        for (n, basis_val) in enumerate(bspline_basis_vals)
            # Multiply basis function value by control point value
            bspline_val += pcof[offset+i-1+n] * basis_val
        end

        vals_vec[k] = bspline_val
    end
    return vals_vec
end
