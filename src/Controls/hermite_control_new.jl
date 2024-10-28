mutable struct HermiteControlNew{N_derivatives, ScalingType} <: AbstractControl
    N_coeff:Int64
    tf::Float64
    N_points::Int64
    region_index::Int64
    left_right_p_vals::Vector{Float64}
    left_right_q_vals::Vector{Float64}
    uint_intermediate_p::Vector{Float64}
    uint_intermediate_q::Vector{Float64}
    vals_working_vec::Vector{Float64}

    function HermiteControlNew(N_points::Integer, tf::Real, 
            N_derivatives::Integer, scaling_type::Symbol=:Heuristic
        )

        N_points = convert(Int64, N_points)
        tf = convert(Float64, tf)
        region_index = -1

        vals_width = N_derivatives+1
        N_coeff = N_points*vals_width*2
        dt = tf / (N_points-1)
        Hmat = zeros(2*vals_width, 2*vals_width)
        # Hermite interpolant will be centered about the middle of each "control timestep"
        xl = 0.0
        xr = 1.0
        xc = 0.5
        icase = 0
        Hermite_map!(Hmat, N_derivatives, xl, xr, xc, icase)

        left_right_p_vals = zeros(2*vals_width)
        left_right_q_vals = zeros(2*vals_width)
        uint_intermediate_p = zeros(2*vals_width)
        uint_intermediate_q = zeros(2*vals_width)
        vals_working_vec= zeros(2*vals_width)


        # Enforce scaling type at construction
        if (scaling_type != :Taylor) && (scaling_type != :Derivative) && (scaling_type != :Heuristic)
            throw(ArgumentError(string(control.scaling_type)))
        end

        new(N_coeff, tf, N_points, region_index, left_right_p_vals, 
            left_right_q_vals, uint_intermediate_p, uint_intermediate_q, vals_working_vec)
    end
end



function update_region!(control::HermiteControlNew{N_derivatives, ScalingType},
        t::Real, pcof::AbstractVector{<: Real}
    ) where {N_derivatives, ScalingType}
    region_index = find_region_index(t, control.tf, control.N_points-1)
    if region_index != control.region_index
        # The number of function values and derivatives specified at each point
        vals_width = N_derivatives+1 

        offset_n_p = 1 + (region_index-1)*vals_width
        offset_n_q = offset_n_p + div(control.N_coeff, 2)
        copyto!(control.left_right_p_vals, 1, pcof, offset_n_p, 2*vals_width)
        copyto!(control.left_right_q_vals, 1, pcof, offset_n_q, 2*vals_width)

        # Try to get scaling so that the parameters corresponding to higher
        # order derivatives have the same impact as the parameters
        # corresponding to lower order derivatives.
        for i in 0:control.N_derivatives
            # The if-statement should get compiled out.
            if (control.scaling_type == :Taylor)
                scaling_factor = 1
            elseif (control.scaling_type == :Derivative)
                scaling_factor = dt^i / factorial(i)
            elseif (control.scaling_type == :Heuristic)
                scaling_factor = factorial(i+1)*2^i
            end

            control.left_right_p_vals[1+i] *= scaling_factor
            control.left_right_p_vals[1+i+val_width] *= scaling_factor
            control.left_right_q_vals[1+i] *= scaling_factor
            control.left_right_q_vals[1+i+val_width] *= scaling_factor

        end

        mul!(control.uint_intermediate_p, control.Hmat, control.left_right_p_vals)
        mul!(control.uint_intermediate_q, control.Hmat, control.left_right_q_vals)
    end

    control.region_index = region_index

    return region_index
end

function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::HermiteControlNew{N_derivatives, Symbol}, t::Real,
        pcof::AbstractVector{<: Real}
    ) where {N_derivatives, Symbol}

    region_index = update_region!(control, t, pcof)

    dt = control.dt

    tn = dt*(i-1)
    t_center = tn + 0.5*dt
    tnp1 = dt*i

    t_normalized = (t - t_center)/dt
    extrapolate2!(control.vals_working_vec, control.uint_intermediate_p, t_normalized)
    copyto!(vals_vec, 1, control.vals_working_vec, length(vals_vec))

    return vals_vec
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::HermiteControlNew{N_derivatives, Symbol}, t::Real,
        pcof::AbstractVector{<: Real}
    )

    region_index = update_region!(control, t, pcof)

    dt = control.dt

    tn = dt*(i-1)
    t_center = tn + 0.5*dt
    tnp1 = dt*i

    t_normalized = (t - t_center)/dt
    extrapolate2!(control.vals_working_vec, control.uint_intermediate_q, t_normalized)
    copyto!(vals_vec, 1, control.vals_working_vec, length(vals_vec))

    return vals_vec
end



# TODO Test the above, implement gradients. I think this can be accomplished
# with another working vector, doing the Hmat multiplication by hand and then
# doing the extrapolation. But something even more efficient could probably be
# hard-coded.
