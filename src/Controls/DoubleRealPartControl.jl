"""
Wrapper which takes outputs 2 times the real part of the base control. Should
be more efficient for lab frame conversion of controls which already have
carrier waves.
"""
struct DoubleRealPartControl{BaseControlT} <: AbstractControl
    base_control::BaseControlT
    N_coeff::Int64
    tf::Float64
end

function DoubleRealPartControl(base_control::AbstractControl)
    return DoubleRealPartControl(
        base_control,
        base_control.N_coeff,
        base_control.tf,
    )
end

function eval_p(control::DoubleRealPartControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end

function eval_q(control::DoubleRealPartControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end

function eval_p_derivative(control::DoubleRealPartControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    return 2 * eval_p_derivative(control.base_control, t, pcof, order)
end

function eval_q_derivative(control::DoubleRealPartControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    return 0.0
end

function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::DoubleRealPartControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    fill_p_vec!(vals_vec, control.base_control, t, pcof)
    vals_vec .*= 2
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::DoubleRealPartControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    vals_vec .*= 0.0
end

function eval_grad_p_derivative!(grad::AbstractVector{<: Real},
        control::DoubleRealPartControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)
    eval_grad_p_derivative!(grad, control.base_control, t, pcof, order)
    grad .*= 2
end

function eval_grad_q_derivative!(grad::AbstractVector{<: Real},
        control::DoubleRealPartControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)
    grad .*= 0
end

function fill_grad_p_mat!(
        grad_mat::AbstractMatrix{Float64}, control::DoubleRealPartControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    fill_grad_p_mat!(grad_mat, control.base_control, t, pcof)
    grad_mat .*= 2
end

function fill_grad_q_mat!(
        grad_mat::AbstractMatrix{Float64}, control::DoubleRealPartControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    grad_mat .= 0
end
