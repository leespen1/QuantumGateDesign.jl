"""
GRAPE-style Control: piecewise constant control 
(unlike GRAPE, number of parameters is independent of number of timesteps,
and we use our methods of time stepping and gradient calculation)

May be a little tricky to handle discontinuities. If the problem stepsize is 
chosen to match the duration of each constant amplitude, then ideally I would
make sure that for each timestep the same amplitude is used at t and t+dt, so
that the timesteps reflect that the amplitude is constant across the entire
interval.

For now, I am just going to assume the number of timesteps taken is far greater
than the number of constant amplitudes, and then the results should be pretty
close. But this is something I should address later on 

(perhaps a keyword argument could take care of this?)
"""
struct GRAPEControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_amplitudes::Int64
    function GRAPEControl(N_amplitudes::Int64, tf::Float64)
        N_coeff = N_amplitudes * 2
        new(N_coeff, tf, N_amplitudes)
    end
end


"""
Given a time interval [0,tf], divided into `N_regions` regions, and a time `t`.
Return the index of the region to which `t` belongs.

(should move this to a different file, if I am going to use it in multiple places)
"""
@inline function find_region_index(t, tf, N_regions)
    # Check that t is in the interval
    if (t < 0) || (t > tf)
        throw(DomainError(t, "Value is outside the interval [0,tf]"))
    end
    # Calculate the width of each region
    region_width = tf / N_regions

    # Find the index of the region to which t belongs
    region_index = min(floor(Int, t / region_width) + 1, N_regions)

    return region_index
end

function eval_p(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    i = find_region_index(t, control.tf, control.N_amplitudes)
    return pcof[i]
end

function eval_q(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    i = control.N_amplitudes + find_region_index(t, control.tf, control.N_amplitudes)
    return pcof[i]
end

#
# Old, hard-coded mehtods for evaluating derivatives and gradients (more efficient, )
#

function eval_pt(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    return 0.0
end

function eval_qt(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    return 0.0
end

function eval_grad_p(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    grad = zeros(control.N_coeff)
    i = find_region_index(t, control.tf, control.N_amplitudes)
    grad[i] = 1
    return grad
end

function eval_grad_q(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    grad = zeros(control.N_coeff)
    i = find_region_index(t, control.tf, control.N_amplitudes)
    grad[control.N_amplitudes + i] = 1
    return grad
end

function eval_grad_pt(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    grad = zeros(control.N_coeff)
    return grad
end

function eval_grad_qt(control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    grad = zeros(control.N_coeff)
    return grad
end
