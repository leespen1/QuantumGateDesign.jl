"""
Like the grape control, but instead of each control parameter controlling a
constant amplitude, they control the coefficient of a monomial
(monomial_order=0 correpsonds to classic GRAPE).
"""
struct GeneralGRAPEControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_amplitudes::Int64
    monomial_order::Int64
    function GeneralGRAPEControl(N_amplitudes::Int64, tf::Float64, monomial_order::Int64)
        N_coeff = N_amplitudes * 2
        new(N_coeff, tf, N_amplitudes, monomial_order)
    end
end

function eval_p(control::GeneralGRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    region_index = find_region_index(t, control.tf, control.N_amplitudes)
    region_width = control.tf / control.N_amplitudes
    region_start = region_width*(region_index - 1)
    local_t = (t - region_start)/region_width

    return pcof[region_index]*(local_t^control.monomial_order)
end


function eval_q(control::GeneralGRAPEControl, t::Real, pcof::AbstractVector{<: Real})
    region_index = find_region_index(t, control.tf, control.N_amplitudes)
    region_width = control.tf / control.N_amplitudes
    region_start = region_width*(region_index - 1)
    local_t = (t - region_start)/region_width

    return pcof[control.N_amplitudes + region_index]*(local_t^control.monomial_order)
end
