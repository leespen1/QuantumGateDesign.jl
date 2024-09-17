"""
Be careful of mutating base_control. Ideally base_control should be immutable
(and it's field also immutable)
"""
struct CarrierControl{T} <: AbstractControl
    base_control::T
    N_coeffs_per_frequency::Int64
    N_coeff::Int64
    tf::Float64
    carrier_frequencies::Vector{Float64}
    pcof_storage::Vector{Float64}
end

function CarrierControl(base_control, carrier_frequencies)
    N_frequencies = length(carrier_frequencies)
    N_coeffs_per_frequency = base_control.N_coeff
    N_coeff = N_coeffs_per_frequency * N_frequencies
    tf = base_control.tf
    pcof_storage = zeros(N_coeffs_per_frequency)
    
    return CarrierControl(base_control, N_coeffs_per_frequency, N_coeff, tf, carrier_frequencies, pcof_storage)
end

function eval_p(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end

function eval_q(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end

"""
This could be made more efficient by storing a vector of the derivatives for
the sin/cos and the base control
"""
function eval_p_derivative(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    val = 0.0
    for (i, w) in enumerate(control.carrier_frequencies)
        offset = (i-1)*control.N_coeffs_per_frequency
        this_carrier_pcof = view(pcof, 1+offset:offset+control.N_coeffs_per_frequency)

        for k in 0:order
            if (k % 4) == 0
                carrier_val1 =  cos(w*t) * (w^k)
                carrier_val2 = -sin(w*t) * (w^k)
            elseif (k % 4) == 1
                carrier_val1 = -sin(w*t) * (w^k)
                carrier_val2 = -cos(w*t) * (w^k)
            elseif (k % 4) == 2
                carrier_val1 = -cos(w*t) * (w^k)
                carrier_val2 =  sin(w*t) * (w^k)
            elseif (k % 4) == 3
                carrier_val1 =  sin(w*t) * (w^k)
                carrier_val2 =  cos(w*t) * (w^k)
            end

            base_val1 = eval_p_derivative(control.base_control, t, this_carrier_pcof, order-k)
            base_val2 = eval_q_derivative(control.base_control, t, this_carrier_pcof, order-k)
            val += binomial(order, k) * ((carrier_val1 * base_val1) + (carrier_val2 * base_val2))
        end 
    end
    return val
end

function eval_q_derivative(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    val = 0.0
    for (i, w) in enumerate(control.carrier_frequencies)
        offset = (i-1)*control.N_coeffs_per_frequency
        this_carrier_pcof = view(pcof, 1+offset:offset+control.N_coeffs_per_frequency)

        for k in 0:order
            if (k % 4) == 0
                carrier_val1 =  sin(w*t) * (w^k)
                carrier_val2 =  cos(w*t) * (w^k)
            elseif (k % 4) == 1
                carrier_val1 =  cos(w*t) * (w^k)
                carrier_val2 = -sin(w*t) * (w^k)
            elseif (k % 4) == 2
                carrier_val1 = -sin(w*t) * (w^k)
                carrier_val2 = -cos(w*t) * (w^k)
            elseif (k % 4) == 3
                carrier_val1 = -cos(w*t) * (w^k)
                carrier_val2 =  sin(w*t) * (w^k)
            end

            base_val1 = eval_p_derivative(control.base_control, t, this_carrier_pcof, order-k)
            base_val2 = eval_q_derivative(control.base_control, t, this_carrier_pcof, order-k)
            val += binomial(order, k) * ((carrier_val1 * base_val1) + (carrier_val2 * base_val2))
        end 
    end
    return val
end

function eval_grad_p_derivative!(grad::AbstractVector{<: Real}, control::CarrierControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    grad .= 0

    for (i, w) in enumerate(control.carrier_frequencies)
        offset = (i-1)*control.N_coeffs_per_frequency
        this_carrier_pcof = view(pcof, 1+offset:offset+control.N_coeffs_per_frequency)
        this_carrier_grad = view(grad, 1+offset:offset+control.N_coeffs_per_frequency)

        for k in 0:order
            if (k % 4) == 0
                carrier_val1 =  cos(w*t) * (w^k)
                carrier_val2 = -sin(w*t) * (w^k)
            elseif (k % 4) == 1
                carrier_val1 = -sin(w*t) * (w^k)
                carrier_val2 = -cos(w*t) * (w^k)
            elseif (k % 4) == 2
                carrier_val1 = -cos(w*t) * (w^k)
                carrier_val2 =  sin(w*t) * (w^k)
            elseif (k % 4) == 3
                carrier_val1 =  sin(w*t) * (w^k)
                carrier_val2 =  cos(w*t) * (w^k)
            end

            control.pcof_storage .= 0
            eval_grad_p_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            @. control.pcof_storage *= carrier_val1 * binomial(order, k)
            this_carrier_grad .+= control.pcof_storage

            control.pcof_storage .= 0
            eval_grad_q_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            @. control.pcof_storage *= carrier_val2 * binomial(order, k)
            this_carrier_grad .+= control.pcof_storage
        end 
    end
    return grad
end
