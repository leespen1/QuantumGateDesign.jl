struct GeneralBSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    bspline_basis::BSplines.BSplineBasis{LinRange{Float64, Int64}}
    storage_vec::Vector{Float64}
end

function GeneralBSplineControl(degree, N_knots, tf)
    order = degree+1
    ts = LinRange(0, tf, N_knots)
    bspline_basis = BSplines.BSplineBasis(order, ts)
    N_coeff = (order + length(ts) - 2)*2
    storage_vec = zeros(order)
    return GeneralBSplineControl(N_coeff, tf, bspline_basis, storage_vec)
end

function eval_p(control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end

function eval_q(control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end

function eval_p_derivative(control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset_storage_vec = BSplines.bsplines!(
        control.storage_vec, control.bspline_basis, t, BSplines.Derivative(order)
    )

    val = 0.0
    for index in eachindex(offset_storage_vec)
        val += offset_storage_vec[index] * pcof[index]
    end
    return val
end

function eval_q_derivative(control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset_storage_vec = BSplines.bsplines!(
        control.storage_vec, control.bspline_basis, t, BSplines.Derivative(order)
    )

    val = 0.0
    coeff_offset = div(control.N_coeff, 2)
    for index in eachindex(offset_storage_vec)
        val += offset_storage_vec[index] * pcof[coeff_offset + index]
    end
    return val
end

"""
Spline is linear in pcof, take advantage of that (as well as the offset indices
of the return value of bsplines!)
"""
function eval_grad_p_derivative!(grad::AbstractVector{Float64}, control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset_storage_vec = BSplines.bsplines!(
        control.storage_vec, control.bspline_basis, t, BSplines.Derivative(order)
    )

    grad .= 0
    for index in eachindex(offset_storage_vec)
        grad[index] = offset_storage_vec[index]
    end

    return grad
end

"""
Spline is linear in pcof, take advantage of that (as well as the offset indices
of the return value of bsplines!)
"""
function eval_grad_q_derivative!(grad::AbstractVector{Float64}, control::GeneralBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    offset_storage_vec = BSplines.bsplines!(
        control.storage_vec, control.bspline_basis, t, BSplines.Derivative(order)
    )

    grad .= 0
    coeff_offset = div(control.N_coeff, 2)
    for index in eachindex(offset_storage_vec)
        grad[coeff_offset+index] = offset_storage_vec[index]
    end

    return grad
end
