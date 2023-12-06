#=================================================
# 
# Bspline/Bcarrier 
#
=================================================#
struct BSplineControl <: AbstractControl
    N_coeff::Int64
    bcpar::bcparams
end

function bspline_control(T::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(T, D1, omega_bcpar, pcof)
    return BSplineControl(bcpar.Ncoeff, bcpar)
end


function eval_p(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2(t, control.bcpar, 1, pcof)
end

function eval_pt(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2_dt(t, control.bcpar, 0, pcof)
end

function eval_qt(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2_dt(t, control.bcpar, 1, pcof)
end

function eval_grad_p(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return gradbcarrier2(t, control.bcpar, 0)
end

function eval_grad_q(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return gradbcarrier2(t, control.bcpar, 1)
end

function eval_grad_pt(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return gradbcarrier2_dt(t, control.bcpar, 0)
end

function eval_grad_qt(control::BSplineControl, t::Real, pcof::AbstractVector{Float64})
    return gradbcarrier2_dt(t, control.bcpar, 1)
end


#=================================================
# 
# Bspline/Bcarrier Autodiff Version (uses automatic differentiation for
# all derivatives)
#
=================================================#

struct BSplineControlAutodiff <: AbstractControl
    N_coeff::Int64
    bcpar::bcparams
end

function bspline_control_autodiff(T::Float64, D1::Int, omega::AbstractVector{Float64})
    pcof = zeros(2*D1*length(omega)) # For now, only doing one coupled pair of control
    omega_bcpar = [omega] # Need to wrap in another vector, since bcparams generally expects multiple controls (multiple frequencies != multiple controls)
    bcpar = bcparams(T, D1, omega_bcpar, pcof)
    return BSplineControlAutodiff(bcpar.Ncoeff, bcpar)
end


function eval_p(control::BSplineControlAutodiff, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2(t, control.bcpar, 0, pcof)
end

function eval_q(control::BSplineControlAutodiff, t::Real, pcof::AbstractVector{Float64})
    return bcarrier2(t, control.bcpar, 1, pcof)
end
