const vals_vec_storage_size::Int64 = 20

"""
Be careful of mutating base_control. Ideally base_control should be immutable
(and it's field also immutable)
"""
struct CarrierControl{T} <: AbstractControl
    base_control::T
    N_coeff::Int64
    tf::Float64
    N_frequencies::Int64
    current_t::Base.RefValue{Float64}
    current_derivative_order::Base.RefValue{Int64}
    carrier_frequencies::Vector{Float64}
    pcof_storage::Vector{Float64}
    base_val_storage::Matrix{Float64} # Each column is a p_vec or q_vec for each of the base controls (with 1/derivative_order! factor)
    carrier_val_storage::Matrix{Float64} # Each column is a p_vec or q_vec for eⁱʷᵗ (with 1/derivative_order! factor)
    function CarrierControl(base_control::AbstractControl, carrier_frequencies::AbstractVector{<: Real})
        carrier_frequencies = convert(Vector{Float64}, carrier_frequencies)

        N_frequencies = length(carrier_frequencies)
        N_coeffs_per_frequency = base_control.N_coeff
        N_coeff = base_control.N_coeff * N_frequencies
        tf = base_control.tf

        current_t = Ref(NaN)
        current_derivative_order = Ref(-1)

        pcof_storage = fill(NaN, N_coeffs_per_frequency)
        base_val_storage = fill(NaN, vals_vec_storage_size, 2*N_frequencies)
        carrier_val_storage = fill(NaN, vals_vec_storage_size, 2*N_frequencies)
        
        new{typeof(base_control)}(
            base_control, N_coeff, tf, N_frequencies, current_t,
            current_derivative_order, carrier_frequencies, pcof_storage,
            base_val_storage, carrier_val_storage
        )
    end
end

@inline function update_carrier_vals!(control::CarrierControl, t::Real, nderiv::Integer)
    updated::Bool = false
    if (t != control.current_t[]) || (nderiv > control.current_derivative_order[])
        control.current_t[] = t
        control.current_derivative_order[] = nderiv

        for (i, w) in enumerate(control.carrier_frequencies)
            offset = (i-1)*2
            w_pow = 1.0
            coswt = cos(w*t)
            sinwt = sin(w*t)
            for k in 0:nderiv
                if (k % 4) == 0
                    carrier_pval =  coswt * w_pow
                    carrier_qval =  sinwt * w_pow
                elseif (k % 4) == 1
                    carrier_pval = -sinwt * w_pow
                    carrier_qval =  coswt * w_pow
                elseif (k % 4) == 2
                    carrier_pval = -coswt * w_pow
                    carrier_qval = -sinwt * w_pow
                elseif (k % 4) == 3
                    carrier_pval =  sinwt * w_pow
                    carrier_qval = -coswt * w_pow
                end
                control.carrier_val_storage[1+k, offset+1] = carrier_pval / factorial(k)
                control.carrier_val_storage[1+k, offset+2] = carrier_qval / factorial(k)
                w_pow *= w
            end
        end
        updated = true
    end
    return updated
end

@inline function update_base_vals!(control::CarrierControl, t::Real, nderiv::Integer, pcof::AbstractVector{<: Real})
    for i in 1:control.N_frequencies
        offset = (i-1)*2
        pcof_offset = (i-1)*control.base_control.N_coeff
        this_carrier_pcof = view(pcof, pcof_offset+1:pcof_offset+control.base_control.N_coeff)

        base_p_vec = view(control.base_val_storage, 1:1+nderiv, 1+offset)
        base_q_vec = view(control.base_val_storage, 1:1+nderiv, 2+offset)

        fill_p_vec!(base_p_vec, control.base_control, t, this_carrier_pcof)
        fill_q_vec!(base_q_vec, control.base_control, t, this_carrier_pcof)
    end
    return nothing
end

function eval_p(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end


function eval_q(control::CarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end


function eval_p_derivative(control::CarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    for i in 1:control.N_frequencies
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+order-k, 1+offset]
            val -= control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+order-k, 2+offset]
        end
    end
    val *= factorial(order)
    return val
end

function eval_q_derivative(control::CarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    for i in 1:control.N_frequencies
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+order-k, 2+offset]
            val += control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+order-k, 1+offset]
        end
    end
    val *= factorial(order)
    return val
end


function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::CarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        for i in 1:control.N_frequencies
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+n-k, 1+offset]
                val -= control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+n-k, 2+offset]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
end


function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::CarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        for i in 1:control.N_frequencies
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+n-k, 2+offset]
                val += control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+n-k, 1+offset]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
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

            binomial_coeff = binomial(order, k)

            control.pcof_storage .= 0
            eval_grad_p_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val1 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage

            control.pcof_storage .= 0
            eval_grad_q_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val2 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage
        end 
    end
    return grad
end

function eval_grad_q_derivative!(grad::AbstractVector{<: Real}, control::CarrierControl{T}, t::Real, pcof::AbstractVector{<: Real}, order::Int64) where T
    grad .= 0

    for (i, w) in enumerate(control.carrier_frequencies)
        offset = (i-1)*control.N_coeffs_per_frequency
        this_carrier_pcof = view(pcof, 1+offset:offset+control.N_coeffs_per_frequency)
        this_carrier_grad = view(grad, 1+offset:offset+control.N_coeffs_per_frequency)

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

            binomial_coeff = binomial(order, k)

            control.pcof_storage .= 0
            eval_grad_p_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val1 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage

            control.pcof_storage .= 0
            eval_grad_q_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val2 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage
        end 
    end
    return grad
end

# WORK IN PROGRESS - Want to use the linearity of the Bsplines to my advantage
function eval_grad_p_derivative!(grad::AbstractVector{<: Real}, control::CarrierControl{FortranBSplineControl2}, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    grad .= 0

    update!(control.base_control.bspline, t/control.tf, order)

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

            binomial_coeff = binomial(order, k)

            control.pcof_storage .= 0
            eval_grad_p_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val1 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage

            control.pcof_storage .= 0
            eval_grad_q_derivative!(
                control.pcof_storage, control.base_control, t,
                this_carrier_pcof, order-k
            )

            control.pcof_storage .*= carrier_val2 * binomial_coeff
            this_carrier_grad .+= control.pcof_storage
        end 
    end
    return grad
end


#=
"""
Be careful of mutating base_control. Ideally base_control should be immutable
(and it's field also immutable)
"""
struct StaticCarrierControl{T,Nfreq,Nfreq2} <: AbstractControl
    base_control::T
    N_coeff::Int64
    tf::Float64
    carrier_frequencies::NTuple{Nfreq, Float64}
    base_val_storage::SVector{Nfreq2, MVector{vals_vec_storage_size, Float64}} # Each column is a p_vec or q_vec for each of the base controls (with 1/derivative_order! factor)
    carrier_val_storage::SVector{Nfreq2, MVector{vals_vec_storage_size, Float64}} # Each column is a p_vec or q_vec for eⁱʷᵗ (with 1/derivative_order! factor)
    current_t::Base.RefValue{Float64}
    current_derivative_order::Base.RefValue{Int64}
    pcof_storage::Vector{Float64}
    function StaticCarrierControl(base_control::AbstractControl, carrier_frequencies::NTuple{Nfreq, Float64}) where Nfreq

        N_frequencies = length(carrier_frequencies)
        N_coeffs_per_frequency = base_control.N_coeff
        N_coeff = base_control.N_coeff * N_frequencies
        tf = base_control.tf

        Nfreq2 = 2*Nfreq
        #base_val_storage = MMatrix{10, Nfreq2, Float64}(undef)
        #carrier_val_storage = MMatrix{10, Nfreq2, Float64}(undef)

        base_val_storage = SVector{Nfreq2, MVector{vals_vec_storage_size, Float64}}(
            [MVector{vals_vec_storage_size, Float64}(fill(NaN, vals_vec_storage_size)) for _ in 1:Nfreq2]...
        )
        carrier_val_storage = SVector{Nfreq2, MVector{vals_vec_storage_size, Float64}}(
            [MVector{vals_vec_storage_size, Float64}(fill(NaN, vals_vec_storage_size)) for _ in 1:Nfreq2]...
        )

        current_t = Ref(NaN)
        current_derivative_order = Ref(-1)

        pcof_storage = fill(NaN, N_coeffs_per_frequency)
        
        new{typeof(base_control), Nfreq, Nfreq2}(
            base_control, N_coeff, tf, carrier_frequencies, 
            base_val_storage, carrier_val_storage, current_t,
            current_derivative_order, pcof_storage
        )
    end
end

@inline function update_carrier_vals!(control::StaticCarrierControl, t::Real, nderiv::Integer)
    updated::Bool = false
    if (t != control.current_t[]) || (nderiv > control.current_derivative_order[])
        control.current_t[] = t
        control.current_derivative_order[] = nderiv

        for (i, w) in enumerate(control.carrier_frequencies)
            offset = (i-1)*2
            w_pow = 1.0
            coswt = cos(w*t)
            sinwt = sin(w*t)
            for k in 0:nderiv
                if (k % 4) == 0
                    carrier_pval =  coswt * w_pow
                    carrier_qval =  sinwt * w_pow
                elseif (k % 4) == 1
                    carrier_pval = -sinwt * w_pow
                    carrier_qval =  coswt * w_pow
                elseif (k % 4) == 2
                    carrier_pval = -coswt * w_pow
                    carrier_qval = -sinwt * w_pow
                elseif (k % 4) == 3
                    carrier_pval =  sinwt * w_pow
                    carrier_qval = -coswt * w_pow
                end
                control.carrier_val_storage[offset+1][1+k] = carrier_pval / factorial(k)
                control.carrier_val_storage[offset+2][1+k] = carrier_qval / factorial(k)
                w_pow *= w
            end
        end
        updated = true
    end
    return updated
end

@inline function update_base_vals!(control::StaticCarrierControl, t::Real, nderiv::Integer, pcof::AbstractVector{<: Real})
    for i in eachindex(control.carrier_frequencies)
        offset = (i-1)*2
        pcof_offset = (i-1)*control.base_control.N_coeff
        this_carrier_pcof = view(pcof, pcof_offset+1:pcof_offset+control.base_control.N_coeff)

        #base_p_vec = view(control.base_val_storage, 1:nderiv, 1+offset)
        #base_q_vec = view(control.base_val_storage, 1:nderiv, 2+offset)
        base_p_vec = control.base_val_storage[1+offset]
        base_q_vec = control.base_val_storage[2+offset]

        fill_p_vec!(base_p_vec, control.base_control, t, this_carrier_pcof)
        fill_q_vec!(base_q_vec, control.base_control, t, this_carrier_pcof)
    end
    return nothing
end

function eval_p(control::StaticCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end


function eval_q(control::StaticCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end


function eval_p_derivative(control::StaticCarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    for i in eachindex(control.carrier_frequencies)
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+offset][1+k] * control.base_val_storage[1+offset][1+order-k]
            val -= control.carrier_val_storage[2+offset][1+k] * control.base_val_storage[2+offset][1+order-k]
        end
    end
    val *= factorial(order)
    return val
end

function eval_q_derivative(control::StaticCarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    for i in eachindex(control.carrier_frequencies)
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+offset][1+k] * control.base_val_storage[2+offset][1+order-k]
            val += control.carrier_val_storage[2+offset][1+k] * control.base_val_storage[1+offset][1+order-k]
        end
    end
    val *= factorial(order)
    return val
end


function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::StaticCarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        for i in eachindex(control.carrier_frequencies)
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+offset][1+k] * control.base_val_storage[1+offset][1+n-k]
                val -= control.carrier_val_storage[2+offset][1+k] * control.base_val_storage[2+offset][1+n-k]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
end


function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::StaticCarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        for i in eachindex(control.carrier_frequencies)
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+offset][1+k] * control.base_val_storage[2+offset][1+n-k]
                val += control.carrier_val_storage[2+offset][1+k] * control.base_val_storage[1+offset][1+n-k]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
end


















"""
Be careful of mutating base_control. Ideally base_control should be immutable
(and it's field also immutable)
"""
struct TurboCarrierControl{T} <: AbstractControl
    base_control::T
    N_coeff::Int64
    tf::Float64
    N_frequencies::Int64
    current_t::Base.RefValue{Float64}
    current_derivative_order::Base.RefValue{Int64}
    carrier_frequencies::Vector{Float64}
    pcof_storage::Vector{Float64}
    base_val_storage::Matrix{Float64} # Each column is a p_vec or q_vec for each of the base controls (with 1/derivative_order! factor)
    carrier_val_storage::Matrix{Float64} # Each column is a p_vec or q_vec for eⁱʷᵗ (with 1/derivative_order! factor)
    function TurboCarrierControl(base_control::AbstractControl, carrier_frequencies::AbstractVector{<: Real})
        carrier_frequencies = convert(Vector{Float64}, carrier_frequencies)

        N_frequencies = length(carrier_frequencies)
        N_coeffs_per_frequency = base_control.N_coeff
        N_coeff = base_control.N_coeff * N_frequencies
        tf = base_control.tf

        current_t = Ref(NaN)
        current_derivative_order = Ref(-1)

        pcof_storage = fill(NaN, N_coeffs_per_frequency)
        base_val_storage = fill(NaN, vals_vec_storage_size, 2*N_frequencies)
        carrier_val_storage = fill(NaN, vals_vec_storage_size, 2*N_frequencies)
        
        new{typeof(base_control)}(
            base_control, N_coeff, tf, N_frequencies, current_t,
            current_derivative_order, carrier_frequencies, pcof_storage,
            base_val_storage, carrier_val_storage
        )
    end
end

@inline function update_carrier_vals!(control::TurboCarrierControl, t::Real, nderiv::Integer)
    @assert nderiv < 19
    updated::Bool = false
    if (t != control.current_t[]) || (nderiv > control.current_derivative_order[])
        control.current_t[] = t
        control.current_derivative_order[] = nderiv

        for (i, w) in enumerate(control.carrier_frequencies)
            offset = (i-1)*2
            w_pow = 1.0
            coswt = cos(w*t)
            sinwt = sin(w*t)
            # TODO Might be worth it to compute w^k each time and use turbo on the loop
            for k in 0:nderiv
                if (k % 4) == 0
                    carrier_pval =  coswt * w_pow
                    carrier_qval =  sinwt * w_pow
                elseif (k % 4) == 1
                    carrier_pval = -sinwt * w_pow
                    carrier_qval =  coswt * w_pow
                elseif (k % 4) == 2
                    carrier_pval = -coswt * w_pow
                    carrier_qval = -sinwt * w_pow
                elseif (k % 4) == 3
                    carrier_pval =  sinwt * w_pow
                    carrier_qval = -coswt * w_pow
                end
                @inbounds control.carrier_val_storage[1+k, offset+1] = carrier_pval / factorial(k)
                @inbounds control.carrier_val_storage[1+k, offset+2] = carrier_qval / factorial(k)
                w_pow *= w
            end
        end
        updated = true
    end
    return updated
end

@inline function update_base_vals!(control::TurboCarrierControl, t::Real, nderiv::Integer, pcof::AbstractVector{<: Real})
    for i in 1:control.N_frequencies
        offset = (i-1)*2
        pcof_offset = (i-1)*control.base_control.N_coeff
        this_carrier_pcof = view(pcof, pcof_offset+1:pcof_offset+control.base_control.N_coeff)

        base_p_vec = view(control.base_val_storage, 1:1+nderiv, 1+offset)
        base_q_vec = view(control.base_val_storage, 1:1+nderiv, 2+offset)

        fill_p_vec!(base_p_vec, control.base_control, t, this_carrier_pcof)
        fill_q_vec!(base_q_vec, control.base_control, t, this_carrier_pcof)
    end
    return nothing
end

function eval_p(control::TurboCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_p_derivative(control, t, pcof, 0)
end


function eval_q(control::TurboCarrierControl, t::Real, pcof::AbstractVector{<: Real})
    return eval_q_derivative(control, t, pcof, 0)
end


function eval_p_derivative(control::TurboCarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_vector_indices_are_1_to_length(pcof)
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    @turbo for i in 1:control.N_frequencies
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+order-k, 1+offset]
            val -= control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+order-k, 2+offset]
        end
    end
    val *= factorial(order)
    return val
end

function eval_q_derivative(control::TurboCarrierControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Int64
    )
    check_vector_indices_are_1_to_length(pcof)
    check_pcof_length(control, pcof)
    update_carrier_vals!(control, t, order)
    update_base_vals!(control, t, order, pcof)

    val = 0.0
    @turbo for i in 1:control.N_frequencies
        offset = (i-1)*2
        for k = 0:order
            val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+order-k, 2+offset]
            val += control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+order-k, 1+offset]
        end
    end
    val *= factorial(order)
    return val
end


function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::TurboCarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_vector_indices_are_1_to_length(vals_vec)
    check_vector_indices_are_1_to_length(pcof)
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        @turbo for i in 1:control.N_frequencies
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+n-k, 1+offset]
                val -= control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+n-k, 2+offset]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
end


function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::TurboCarrierControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    check_vector_indices_are_1_to_length(vals_vec)
    check_vector_indices_are_1_to_length(pcof)
    check_pcof_length(control, pcof)
    nderiv = length(vals_vec) - 1
    update_carrier_vals!(control, t, nderiv)
    update_base_vals!(control, t, nderiv, pcof)

    for n = 0:nderiv
        val = 0.0
        @turbo for i in 1:control.N_frequencies
            offset = (i-1)*2
            for k = 0:n
                val += control.carrier_val_storage[1+k, 1+offset] * control.base_val_storage[1+n-k, 2+offset]
                val += control.carrier_val_storage[1+k, 2+offset] * control.base_val_storage[1+n-k, 1+offset]
            end
        end
        vals_vec[1+n] = val
    end
    return vals_vec
end
=#
