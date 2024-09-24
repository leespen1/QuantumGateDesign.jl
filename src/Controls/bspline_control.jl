#=================================================
# 
# Bspline/Bcarrier 
#
=================================================#
"""
    spar = splineparams(T, D1, Nseg, pcof)

Constructor for struct splineparams, which sets up the parameters for a regular B-spline function
(without carrier waves).

# Arguments
- `T:: Float64`: Duration of spline function
- `D1:: Int64`: Number of basis functions in each spline
- `Nseg:: Int64`:  Number of splines (real, imaginary, different ctrl func)
- `pcof:: Array{Float64, 1}`: Coefficient vector. Must have D1*Nseg elements

# External links
* [Spline Wavelet](https://en.wikipedia.org/wiki/Spline_wavelet#Quadratic_B-spline) on Wikipedia.
"""
struct MySplineControl 
    N_coeff:: Int64 # Total number of coefficients
    tf::Float64
    D1::Int64 # Number of coefficients per spline (e.g. per control function, and in our case we have 2, p and q)
    Nseg::Int64 # Number of segments (real, imaginary, different ctrl func)
    tcenter::Vector{Float64}
    dtknot::Float64

# new, simplified constructor
    function MySplineControl(tf, D1)
        if (D1 < 3)
            throw(ArgumentError("Number of coefficients per spline (D1 = $D1) must be ≥ 3."))
        end

        Nseg = 2 # real and imaginary
        N_coeff = D1*Nseg

        dtknot = tf/(D1 -2)
        tcenter = dtknot.*(collect(1:D1) .- 1.5)

        new(N_coeff, tf, D1, Nseg, tcenter, dtknot)
    end
end

"""
I should make a general function like this in CarrierControl.jl
"""
function my_bspline_controls(tf::Float64, D1::Int, omega::AbstractMatrix{Float64})
    N_controls = size(omega, 1)
    N_freq = size(omega, 2)
    base_control = MySplineControl(tf, D1)
    controls = Vector{CarrierControl{MySplineControl}}()
    for i in 1:N_controls 
        omega_vec = omega[i,:]

        push!(controls, CarrierControl(base_control, omega_vec))
    end

    return controls
end


function eval_p_derivative(
        control::MySplineControl,
        t::Real,
        pcof::AbstractVector{<: Real},
        derivative_order::Integer,
    )
    return bspline2(
        control, t, view(pcof, 1:control.D1),
        derivative_order
    )
end

function eval_q_derivative(
        control::MySplineControl,
        t::Real,
        pcof::AbstractVector{<: Real},
        derivative_order::Integer,
    )
    return bspline2(
        control, t, view(pcof, 1+control.D1:control.N_coeff),
        derivative_order
    )
end

function eval_p(
        control::MySplineControl, t::Real, pcof::AbstractVector{<: Real}
    )
    derivative_order = 0
    return eval_p_derivative(control, t, pcof, derivative_order)
end

function eval_q(
        control::MySplineControl, t::Real, pcof::AbstractVector{<: Real}
    )
    derivative_order = 0
    return eval_q_derivative(control, t, pcof, derivative_order)
end




# bspline2: Evaluate quadratic bspline function
"""
    f = bspline2(t, splineparam, splinefunc)

Evaluate a B-spline function. See also the `splineparams` constructor.

# Arguments
- `t::Real`: Evaluate spline at parameter t ∈ [0, param.T]
- `param::splineparams`: Parameters for the spline
- `splinefunc::Int64`: Spline function index ∈ [0, param.Nseg-1]
"""
@inline function bspline2(
        control::MySplineControl,
        t::Real,
        pcof::AbstractVector{<: Real},
        derivative_order::Integer,
    )

    f = 0.0

    dtknot = control.dtknot
    width = 3*dtknot

    k = max(3, ceil(Int64, (t/dtknot) + 2)) # Unsure if this line does what it is supposed to
    k = min(k, control.D1)

    if (derivative_order == 0)
        # 1st segment of nurb k
        tc = control.tcenter[k]
        tau = (t - tc) / width
        f += pcof[k] * (9/8 + 4.5*tau + 4.5*tau^2) # test to remove square for extra speed

        # 2nd segment of nurb k-1
        tc = control.tcenter[k-1]
        tau = (t - tc) / width
        f += pcof[k-1] * (0.75 - 9*tau^2)

        # 3rd segment of nurb k-2
        tc = control.tcenter[k-2]
        tau = (t - tc) / width
        f += pcof[k-2] * (9/8 - 4.5*tau + 4.5*tau^2)
    elseif (derivative_order == 1)
        # 1st segment of nurb k
        tc = control.tcenter[k]
        tau = (t - tc) / width
        f += pcof[k] * (4.5 + 9*tau) / width # test to remove square for extra speed

        # 2nd segment of nurb k-1
        tc = control.tcenter[k-1]
        tau = (t - tc) / width
        f += pcof[k-1] * (-18*tau) / width

        # 3rd segment of nurb k-2
        tc = control.tcenter[k-2]
        tau = (t - tc) / width
        f += pcof[k-2] * (-4.5 + 9*tau) / width
    elseif (derivative_order == 2)
        # 1st segment of nurb k
        tc = control.tcenter[k]
        tau = (t - tc) / width
        f += pcof[k] * 9 / width^2 # test to remove square for extra speed

        # 2nd segment of nurb k-1
        tc = control.tcenter[k-1]
        tau = (t - tc) / width
        f += pcof[k-1] * -18 / width^2

        # 3rd segment of nurb k-2
        tc = control.tcenter[k-2]
        tau = (t - tc) / width
        f += pcof[k-2] * 9 / width^2
    end
    # If derivative order higher than 2, value is zero

    return f
end

##############################################################################
#
# Old version
# Uses Bcarriers. Now I am just going to use Bsplines with the CarrierControl
# interface.
#
##############################################################################

struct BSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    bcpar::bcparams
end

"""
    BSplineControl(tf, D1, omega)

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

function eval_grad_p_derivative!(grad::AbstractVector{<: Real}, control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    grad .= eval_grad_p_derivative(control, t, pcof, order)
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

function eval_grad_q_derivative!(grad::AbstractVector{<: Real}, control::BSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    grad .= eval_grad_q_derivative(control, t, pcof, order)
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
