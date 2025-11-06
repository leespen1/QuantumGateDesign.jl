"""
Lab frame control, rotated out of the rotating frame. Implemented as a carrier
control, with only the real part taken (and multiplied by two).

When the rotating frame control is a carrier control, this could be done more
efficiently by changing the carrier wave frequencies directly (but would still
need a wrapper to take the 2*Re(⋯ ) ).
"""
struct LabFrameControl{BaseControlT} <: AbstractControl
    lab_carrier_control::CarrierControl{BaseControlT}
    N_coeff::Int64
    tf::Float64
    rotating_frequency::Float64
end

function LabFrameControl(base_control::AbstractControl, rotating_frequency::Real)
    lab_carrier_control = CarrierControl(base_control, [rotating_frequency])
    return LabFrameControl(
        lab_carrier_control,
        base_control.N_coeff,
        base_control.tf,
        rotating_frequency,
    )
end

function eval_p(control::LabFrameControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end

function eval_q(control::LabFrameControl, t::Real, pcof::AbstractVector{<: Real})
    return 0.0
end

function eval_p_derivative(control::LabFrameControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    return 2 * eval_p_derivative(control.lab_carrier_control, t, pcof, order)
end

function eval_q_derivative(control::LabFrameControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    return 0.0
end

function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::LabFrameControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    fill_p_vec!(vals_vec, control.lab_carrier_control, t, pcof)
    vals_vec .*= 2
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::LabFrameControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    vals_vec .*= 0.0
end

function eval_grad_p_derivative!(grad::AbstractVector{<: Real},
        control::LabFrameControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)
    eval_grad_p_derivative!(grad, control.lab_carrier_control, t, pcof, order)
    grad .*= 2
end

function eval_grad_q_derivative!(grad::AbstractVector{<: Real},
        control::LabFrameControl, t::Real, pcof::AbstractVector{<: Real},
        order::Int64)
    grad .*= 0
end

function fill_grad_p_mat!(
        grad_mat::AbstractMatrix{Float64}, control::CarrierControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    fill_grad_p_mat!(grad_mat, control.lab_carrier_control, t, pcof)
    grad_mat .*= 2
end

function fill_grad_q_mat!(
        grad_mat::AbstractMatrix{Float64}, control::CarrierControl, t::Real,
        pcof::AbstractVector{<: Real}
    )
    grad_mat .= 0
end
