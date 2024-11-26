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


function eval_p_derivative(control::GRAPEControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )

    if (order > 0)
        return 0.0
    else
        i = find_region_index(control, t)
        return pcof[i]
    end
end

function eval_q_derivative(control::GRAPEControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )

    if (order > 0)
        return 0.0
    else
        i = find_region_index(control, t)
        offset = control.N_amplitudes
        return pcof[i+offset]
    end
end

function eval_grad_p_derivative!(grad::AbstractVector{<: Real},
        control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real},
        order::Integer
    )
    grad .= 0
    if (order == 0)
        i = find_region_index(control, t)
        grad[i] = 1
    end


    return grad
end

function eval_grad_q_derivative!(grad::AbstractVector{<: Real},
        control::GRAPEControl, t::Real, pcof::AbstractVector{<: Real},
        order::Integer
    )
    grad .= 0
    if (order == 0)
        i = find_region_index(control, t)
        offset = control.N_amplitudes
        grad[i+offset] = 1
    end
    return grad
end

"""
Given a time interval [0,tf], divided into `N_regions` regions, and a time `t`.
Return the index of the region to which `t` belongs.
"""
@inline function find_region_index(control::GRAPEControl, t)
    N_regions = control.N_amplitudes
    tf = control.tf
    # Check that t is in the interval
    if (t < 0) || (t > tf*(1+eps()))
        throw(DomainError(t, "Value is outside the interval [0,tf]"))
    end
    # Calculate the width of each region
    region_width = tf / N_regions

    # Find the index of the region to which t belongs
    region_index = min(floor(Int, t / region_width) + 1, N_regions)

    return region_index
end
