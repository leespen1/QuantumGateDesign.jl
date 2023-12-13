#=================================================
# 
# Bspline/Bcarrier 
#
=================================================#
struct BSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    bcpar::bcparams
end

function bspline_control(tf::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(tf, D1, omega_bcpar, pcof)
    return BSplineControl(bcpar.Ncoeff, tf, bcpar)
end


function eval_p(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 1, pcof)
end

function eval_pt(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2_dt(t, control.bcpar, 0, pcof)
end

function eval_qt(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2_dt(t, control.bcpar, 1, pcof)
end

function eval_grad_p(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return gradbcarrier2(t, control.bcpar, 0)
end

function eval_grad_q(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return gradbcarrier2(t, control.bcpar, 1)
end

function eval_grad_pt(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return gradbcarrier2_dt(t, control.bcpar, 0)
end

function eval_grad_qt(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real})
    return gradbcarrier2_dt(t, control.bcpar, 1)
end

function eval_p_derivative(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    if (order == 0)
        return eval_p(control, t, pcof)
    elseif (order == 1)
        return eval_pt(control, t, pcof)
    else
        throw("Order $order too high")
    end

    return NaN
end

function eval_q_derivative(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    if (order == 0)
        return eval_q(control, t, pcof)
    elseif (order == 1)
        return eval_qt(control, t, pcof)
    else
        throw("Order $order too high")
    end

    return NaN
end

function eval_grad_p_derivative(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    if (order == 0)
        return eval_grad_p(control, t, pcof)
    elseif (order == 1)
        return eval_grad_pt(control, t, pcof)
    else
        throw("Order $order too high")
    end

    return NaN
end

function eval_grad_q_derivative(control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    if (order == 0)
        return eval_grad_q(control, t, pcof)
    elseif (order == 1)
        return eval_grad_qt(control, t, pcof)
    else
        throw("Order $order too high")
    end

    return NaN
end

#=================================================
# 
# Bspline/Bcarrier Autodiff Version 
# (uses automatic differentiation for all derivatives)
#
=================================================#

struct BSplineControlAutodiff <: AbstractControl
    N_coeff::Int64
    tf::Float64
    bcpar::bcparams
end

function bspline_control_autodiff(tf::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(tf, D1, omega_bcpar, pcof)
    return BSplineControlAutodiff(bcpar.Ncoeff, tf, bcpar)
end


function eval_p(control::BSplineControlAutodiff, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControlAutodiff, t::Real, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 1, pcof)
end
