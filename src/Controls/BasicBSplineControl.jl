"""
The first version will assume no repeating knots. You just give the number of
knots / segments.
"""
struct BasicBSplineControl{degree} <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_knots::Int64
    P::BasicBSpline.BSplineSpace{degree, Float64, BasicBSpline.KnotVector{Float64}}
    function BasicBSplineControl(degree::Integer, tf::Real, N_knots::Integer)
        degree = convert(Int64, degree)
        tf = convert(Float64, tf)
        N_knots = convert(Int64, N_knots)

        N_coeff = 2*N_knots

        # Make the knots on the interval [0,1], then scale the inputs t when evaluating
        knot_vector = BasicBSpline.KnotVector(LinRange(0, 1, N_knots))
        P = BSplineSpace{degree}(knot_vector)

        new{degree}(N_coeff, tf, N_knots, P)
    end
end

function eval_p(control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real})
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(control.P, i, t_scaled)
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
    bspline_basis_vals = BasicBSpline.bsplinebasisall(control.P, i, t_scaled)
    for (n, basis_val) in enumerate(bspline_basis_vals)
        # Multiply basis function value by control point value
        bspline_val += pcof[offset+i-1+n] * basis_val
    end

    return bspline_val
end

function eval_p_derivative(
        control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real},
        drv::Integer
    )
    return eval_p_derivative(control, t, pcof, Derivative(drv))
end

function eval_p_derivative(
        control::BasicBSplineControl, t::Real, pcof::AbstractVector{<: Real},
        drv::Derivative{order}
    ) where order
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_drv = BSplineDerivativeSpace{order}(control.P)

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(P_drv, i, t_scaled)
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
    offset = div(control.N_coeff, 2)

    bspline_val = 0.0
    bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[1+order], i, t_scaled)
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
        pcof::AbstractVector{<: Real}, max_drv::Derivative{N}
    ) where N

    drvs = map(QuantumGateDesign.Derivative, ntuple(i -> i-1, N+1))
    drv_vals = map(drv -> eval_p_derivative(control, t, pcof, drv), drvs)
    copyto!(vals_vec, 1, drv_vals, 1, min(length(vals_vec), length(drv_vals)))

    return vals_vec
    #=
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_coeff, 2)

    for k in eachindex(vals_vec)
        bspline_val = 0.0
        bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[k], i, t_scaled)

        for (n, basis_val) in enumerate(bspline_basis_vals)
            # Multiply basis function value by control point value
            bspline_val += pcof[i-1+n] * basis_val
        end

        vals_vec[k] = bspline_val
    end
    return vals_vec
    =#
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::BasicBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}, max_drv::Derivative{N}
    ) where N

    drvs = map(QuantumGateDesign.Derivative, ntuple(i -> i-1, N+1))
    drv_vals = map(drv -> eval_q_derivative(control, t, pcof, drv), drvs)
    copyto!(vals_vec, 1, drv_vals, 1, min(length(vals_vec), length(drv_vals)))

    return vals_vec
    #=
    # BSpline is scaled on [0,1], rescale t accordingly
    t_scaled = t / control.tf

    # Get BSplineSpace and derivatives
    P_tup = (control.P, control.P1, control.P2, control.P3, control.P4)

    # Get the index of the first BSpline Basis function with support at t
    i = BasicBSpline.intervalindex(control.P, t_scaled)
    offset = div(control.N_coeff, 2)

    for k in eachindex(vals_vec)
        bspline_val = 0.0
        bspline_basis_vals = BasicBSpline.bsplinebasisall(P_tup[k], i, t_scaled)

        for (n, basis_val) in enumerate(bspline_basis_vals)
            # Multiply basis function value by control point value
            bspline_val += pcof[offset+i-1+n] * basis_val
        end

        vals_vec[k] = bspline_val
    end
    return vals_vec
    =#
end
