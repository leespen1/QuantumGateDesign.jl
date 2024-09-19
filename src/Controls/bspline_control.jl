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

"""
    BsplineControl(tf, D1, omega)

Construct a control whose value is the sum of Bspline envelopes multiplied by carrier waves.
"""
function BSplineControl(tf::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(tf, D1, omega_bcpar, pcof)
    return BSplineControl(bcpar.Ncoeff, tf, bcpar)
end

"""
Whereas juqbox uses one struct for multiple operators, here I use one struct
per operator.
"""
function bspline_controls(tf::Float64, D1::Int, omega::AbstractMatrix{Float64})
    N_controls = size(omega, 1)
    N_freq = size(omega, 2)
    controls = Vector{BSplineControl}()
    for i in 1:N_controls 
        omega_vec = omega[i,:]
        push!(controls, BSplineControl(tf, D1, omega_vec))
    end

    return controls
end


function eval_p(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return bcarrier2(t, control.bcpar, 1, pcof)
end

function eval_pt(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return bcarrier2_dt(t, control.bcpar, 0, pcof)
end

function eval_qt(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return bcarrier2_dt(t, control.bcpar, 1, pcof)
end

function eval_grad_p(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return gradbcarrier2(t, control.bcpar, 0)
end

function eval_grad_q(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return gradbcarrier2(t, control.bcpar, 1)
end

function eval_grad_pt(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return gradbcarrier2_dt(t, control.bcpar, 0)
end

function eval_grad_qt(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real})
    return gradbcarrier2_dt(t, control.bcpar, 1)
end

function eval_p_derivative(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real}, order::Int64)::Float64
    if (order == 0)
        return eval_p(control, t, pcof)
    elseif (order == 1)
        return eval_pt(control, t, pcof)
    else
        throw("Derivative order $order too high for this control")
    end

    return NaN
end

function eval_q_derivative(control::BSplineControl, t::Float64, pcof::AbstractVector{<: Real}, order::Int64)::Float64
    if (order == 0)
        return eval_q(control, t, pcof)
    elseif (order == 1)
        return eval_qt(control, t, pcof)
    else
        throw("Derivative order $order too high for this control")
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

function BSplineControlAutodiff(tf::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(tf, D1, omega_bcpar, pcof)
    return BSplineControlAutodiff(bcpar.Ncoeff, tf, bcpar)
end

"""
Whereas juqbox uses one struct for multiple operators, here I use one struct
per operator.
"""
function bspline_controls_autodiff(tf::Float64, D1::Int, omega::AbstractMatrix{Float64})
    N_controls = size(omega, 1)
    N_freq = size(omega, 2)
    controls = Vector{BSplineControlAutodiff}()
    for i in 1:N_controls 
        omega_vec = omega[i,:]
        push!(controls, BSplineControlAutodiff(tf, D1, omega_vec))
    end

    return controls
end


function eval_p(control::BSplineControlAutodiff, t::T, pcof::AbstractVector{<: Real})::T where T <: Real
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControlAutodiff, t::T, pcof::AbstractVector{<: Real})::T where T <: Real
    return bcarrier2(t, control.bcpar, 1, pcof)
end
