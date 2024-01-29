"""
Working on arbitrary order version of utvt!, uttvtt!, etc.

uv_matrix's first column is u and v stacked. Second column is ut and vt stacked, etc.

u and v should be given, and the derivatives are to be computed in place by
this method.

"""
function arbitrary_order_uv_derivative!(uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64; use_adjoint::Bool=false,
        forcing_matrix::Union{AbstractMatrix{Float64}, Missing}=missing
    )

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end

    adjoint_factor = use_adjoint ? -1 : 1

    for j = 0:(N_derivatives-1)
        # Get views of the current derivative we are trying to compute (the j+1th derivative)
        u_derivative = view(uv_matrix, 1:prob.N_tot_levels,                       1+j+1)
        v_derivative = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j+1)

        u_derivative .= 0
        v_derivative .= 0

        # Get views of one of the previous derivatives (at first, the derivative just before the current one)
        u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+j)
        v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j)

        # In (15), only i=j has the system operators present, others are only
        # control operators. So system operators are handled outside the loop.
        mul!(u_derivative, prob.system_asym, u_derivative_prev, adjoint_factor, 1)
        mul!(u_derivative, prob.system_sym,  v_derivative_prev, adjoint_factor, 1)

        mul!(v_derivative, prob.system_asym, v_derivative_prev, adjoint_factor, 1)
        mul!(v_derivative, prob.system_sym,  u_derivative_prev, -adjoint_factor, 1)


        # Perform the summation (the above is part of the i=j term in summation, this loop completes that term and the rest)
        for i = j:-1:0
            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            for k in 1:prob.N_operators
                control = controls[k]
                sym_op = prob.sym_operators[k]
                asym_op = prob.asym_operators[k]
                this_pcof = get_control_vector_slice(pcof, controls, k)

                p_val = eval_p_derivative(control, t, this_pcof, j-i) / factorial(j-i)
                q_val = eval_q_derivative(control, t, this_pcof, j-i) / factorial(j-i)

                mul!(u_derivative, asym_op, u_derivative_prev, adjoint_factor*q_val, 1)
                mul!(u_derivative, sym_op,  v_derivative_prev, adjoint_factor*p_val, 1)

                mul!(v_derivative, asym_op, v_derivative_prev, adjoint_factor*q_val,  1)
                mul!(v_derivative, sym_op,  u_derivative_prev, -adjoint_factor*p_val, 1)
            end
        end

        # I believe checking like this means that if-block will be compiled out when no forcing matrix is given
        if !ismissing(forcing_matrix)
            axpy!(1.0, view(forcing_matrix, 1:prob.N_tot_levels,                       1+j), u_derivative)
            axpy!(1.0, view(forcing_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j), v_derivative)
        end

        mul!(u_derivative, u_derivative, 1/(j+1))
        mul!(v_derivative, v_derivative, 1/(j+1))
    end
    #println("\n")
    
    return nothing
end

"""
lambda_in = Λ₀

Returns Λ_(derivative_index)

Need to describe the recursion better.
"""
function arbitrary_order_adjoint_derivative_single(lambda_in::AbstractVector{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        derivative_index::Int64; 
    )

    adjoint_factor = -1.0

    if (derivative_index == 0)
        return lambda_in
    end

    lambda_out = zeros(length(lambda_in))

    lambda_in_u = view(lambda_in, 1:prob.N_tot_levels)
    lambda_in_v = view(lambda_in, 1+prob.N_tot_levels:prob.real_system_size)

    lambda_in_temp = copy(lambda_in)
    lambda_in_temp_u = view(lambda_in_temp, 1:prob.N_tot_levels)
    lambda_in_temp_v = view(lambda_in_temp, 1+prob.N_tot_levels:prob.real_system_size)

    j = derivative_index
    for i=(derivative_index-1):-1:0

        lambda_in_temp .= 0
        
        # Apply A_(derivative_i) to Λ₀, to get our "new Λ₀"
        
        # Apply system operators if we haven't taken any time derivatives
        if (i == 0)
            mul!(lambda_in_temp_u, prob.system_asym, lambda_in_u, adjoint_factor, 1)
            mul!(lambda_in_temp_u, prob.system_sym,  lambda_in_v, adjoint_factor, 1)

            mul!(lambda_in_temp_v, prob.system_asym, lambda_in_v, adjoint_factor, 1)
            mul!(lambda_in_temp_v, prob.system_sym,  lambda_in_u, -adjoint_factor, 1)
        end

        for k in 1:prob.N_operators
            control = controls[k]
            sym_op = prob.sym_operators[k]
            asym_op = prob.asym_operators[k]
            this_pcof = get_control_vector_slice(pcof, controls, k)

            p_val = eval_p_derivative(control, t, this_pcof, i) / factorial(i)
            q_val = eval_q_derivative(control, t, this_pcof, i) / factorial(i)

            mul!(lambda_in_temp_u, asym_op, lambda_in_u, adjoint_factor*q_val, 1)
            mul!(lambda_in_temp_u, sym_op,  lambda_in_v, adjoint_factor*p_val, 1)

            mul!(lambda_in_temp_v, asym_op, lambda_in_v, adjoint_factor*q_val,  1)
            mul!(lambda_in_temp_v, sym_op,  lambda_in_u, -adjoint_factor*p_val, 1)
        end

        lambda_out .+= arbitrary_order_adjoint_derivative_single(
            lambda_in_temp, prob, controls, t, pcof, (derivative_index-1)-i
        )
    end

    return lambda_out ./ derivative_index
end

"""
WIP : Fixing the math, can't just add an adjoint factor into uv_derivative
"""
function arbitrary_order_adjoint_derivative!(
        uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64; 
    )

    lambda_in = uv_matrix[:,1]
    for derivative_i in 1:N_derivatives
        uv_matrix[:,1+derivative_i] .= arbitrary_order_adjoint_derivative_single(
            lambda_in, prob, controls, t, pcof, derivative_i
        )
    end

    return nothing
end



"""
For use in discrete adjoint (maybe it will also be useful in forced gradient)

uv_matrix should already be filled.

The first column of uv_partial_matrix should be zeros.

Maybe forcing matrix should be named forcing_partial_matrix. Because although
it functions the same as before, it is a different thing mathematically.


"""
function arbitrary_order_uv_partial_derivative!(
        uv_partial_matrix::AbstractMatrix{Float64}, uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64, global_pcof_index::Int64
    )

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end


    uv_partial_matrix[:,1] .= 0 # ∂w/∂θₖ = 0

    for j = 0:(N_derivatives-1)
        # Get views of the current derivative we are trying to compute (the j+1th derivative)
        u_partial_derivative = view(uv_partial_matrix, 1:prob.N_tot_levels,                       1+j+1)
        v_partial_derivative = view(uv_partial_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j+1)

        u_partial_derivative .= 0
        v_partial_derivative .= 0

        # Get views of one of the previous derivatives (at first, the derivative just before the current one)
        u_partial_derivative_prev = view(uv_partial_matrix, 1:prob.N_tot_levels,                       1+j)
        v_partial_derivative_prev = view(uv_partial_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j)

        # In (15), only i=j has the system operators present, others are only
        # control operators. So system operators are handled outside the loop.
        mul!(u_partial_derivative, prob.system_asym, u_partial_derivative_prev, 1, 1)
        mul!(u_partial_derivative, prob.system_sym,  v_partial_derivative_prev, 1, 1)

        mul!(v_partial_derivative, prob.system_asym, v_partial_derivative_prev, 1, 1)
        mul!(v_partial_derivative, prob.system_sym,  u_partial_derivative_prev, -1, 1)


        # Perform the summation (the above is part of the i=j term in summation, this loop completes that term and the rest)
        for i = j:-1:0
            # A(∂w/∂θₖ) part
            u_partial_derivative_prev = view(uv_partial_matrix, 1:prob.N_tot_levels,                       1+i)
            v_partial_derivative_prev = view(uv_partial_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            for k in 1:prob.N_operators
                control = controls[k]
                sym_op = prob.sym_operators[k]
                asym_op = prob.asym_operators[k]
                this_pcof = get_control_vector_slice(pcof, controls, k)

                p_val = eval_p_derivative(control, t, this_pcof, j-i) / factorial(j-i)
                q_val = eval_q_derivative(control, t, this_pcof, j-i) / factorial(j-i)

                mul!(u_partial_derivative, asym_op, u_partial_derivative_prev, q_val, 1)
                mul!(u_partial_derivative, sym_op,  v_partial_derivative_prev, p_val, 1)

                mul!(v_partial_derivative, asym_op, v_partial_derivative_prev, q_val,  1)
                mul!(v_partial_derivative, sym_op,  u_partial_derivative_prev, -p_val, 1)
            end

            # (∂A/∂θₖ)w part (only involves one control)
            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            control_index, local_index = get_local_control_indices(controls, global_pcof_index)
            
            local_control = controls[control_index]
            local_pcof = get_control_vector_slice(pcof, controls, control_index)
            sym_op = prob.sym_operators[control_index]
            asym_op = prob.asym_operators[control_index]

            p_val = eval_grad_p_derivative(local_control, t, local_pcof, j-i)[local_index] / factorial(j-i) # Removing the factorial worsened the agreement.
            q_val = eval_grad_q_derivative(local_control, t, local_pcof, j-i)[local_index] / factorial(j-i)

            mul!(u_partial_derivative, asym_op, u_derivative_prev, q_val, 1)
            mul!(u_partial_derivative, sym_op,  v_derivative_prev, p_val, 1)

            mul!(v_partial_derivative, asym_op, v_derivative_prev, q_val,  1)
            mul!(v_partial_derivative, sym_op,  u_derivative_prev, -p_val, 1)
        end

        # Pretty sure I do want the 1/j+1 in this one (only 2nd order agrees if I take it out)
        mul!(u_partial_derivative, u_partial_derivative, 1/(j+1))
        mul!(v_partial_derivative, v_partial_derivative, 1/(j+1))
    end
    #println("\n")
    
    return nothing
end

"""
Non-BLAS Version (so that automatic differentiation works on it)

Turns out it's even tougher, I can't have functions that mutate the inputs
(maybe mutating arrays allocated in the function itself is OK)
"""
function arbitrary_order_uv_derivative_noBLAS(uv_in::AbstractVector{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64; use_adjoint::Bool=false,
        forcing_matrix::Union{AbstractMatrix{Float64}, Missing}=missing
    )

    uv_matrix = zeros(prob.real_system_size, 1+N_derivatives)
    uv_matrix[:,1] .= uv_in

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end

    adjoint_factor = use_adjoint ? -1 : 1


    for j = 0:(N_derivatives-1)
        # Get views of the current derivative we are trying to compute (the j+1th derivative)

        u_derivative = zeros(prob.N_tot_levels)
        v_derivative = zeros(prob.N_tot_levels)

        # Get views of one of the previous derivatives (at first, the derivative just before the current one)
        u_derivative_prev = uv_matrix[1:prob.N_tot_levels,                       1+j]
        v_derivative_prev = uv_matrix[prob.N_tot_levels+1:prob.real_system_size, 1+j]

        # In (15), only i=j has the system operators present, others are only
        # control operators. So system operators are handled outside the loop.
        u_derivative += (prob.system_asym*u_derivative_prev)*adjoint_factor
        u_derivative += (prob.system_sym*v_derivative_prev)*adjoint_factor

        v_derivative += (prob.system_asym*v_derivative_prev)*adjoint_factor
        v_derivative -= (prob.system_sym*u_derivative_prev)*adjoint_factor


        # Perform the summation (the above is part of the i=j term in summation, this loop completes that term and the rest)
        for i = j:-1:0
            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            for k in 1:prob.N_operators
                control = controls[k]
                sym_op = prob.sym_operators[k]
                asym_op = prob.asym_operators[k]
                this_pcof = get_control_vector_slice(pcof, controls, k)

                p_val = eval_p_derivative(control, t, this_pcof, j-i) / factorial(j-i)
                q_val = eval_q_derivative(control, t, this_pcof, j-i) / factorial(j-i)

                u_derivative += (asym_op*u_derivative_prev)*adjoint_factor*q_val
                u_derivative += (sym_op*v_derivative_prev)*adjoint_factor*p_val

                v_derivative += (asym_op*v_derivative_prev)*adjoint_factor*q_val
                v_derivative -= (sym_op*u_derivative_prev)*adjoint_factor*p_val
            end
        end


        u_derivative ./= (j+1)
        v_derivative ./= (j+1)

        uv_matrix[1:prob.N_tot_levels,                       1+j+1] = u_derivative
        uv_matrix[prob.N_tot_levels+1:prob.real_system_size, 1+j+1] = v_derivative

        # I believe checking like this means that if-block will be compiled out when no forcing matrix is given
        if !ismissing(forcing_matrix)
            uv_matrix[:, 1+j+1] += forcing_matrix[1:prob.real_system_size, 1+j]
        end
    end
    #println("\n")
    
    return nothing
end
"""
Apply control to uv_in, ADD result to uv_out
"""
function apply_control_additive!(u_out, v_out, u_in, v_in, control, sym_op, asym_op, pcof, mult_factor, derivative_i)

    p_val = eval_p_derivative(control, t, this_pcof, derivative_i) * mult_factor
    q_val = eval_q_derivative(control, t, this_pcof, derivative_i) * mult_factor

    mul!(u_out, asym_op, u_in, q_val, 1)
    mul!(u_out, sym_op,  v_in, p_val, 1)

    mul!(v_out, asym_op, v_in, q_val,  1)
    mul!(v_out, sym_op,  u_in, -p_val, 1)
end


function coefficient(j,p,q)
    return factorial(p)*factorial(p+q-j)/(factorial(p+q)*factorial(p-j))
end

"""
Compute the RHS/LHS, assuming p=q=N_derivatives
"""
function arbitrary_RHS!(RHS::AbstractVector{Float64}, uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = length(RHS)
    @assert system_size == size(uv_matrix, 1)

    RHS .= 0.0
    for j in 0:N_derivatives
        RHS .+= coefficient(j,N_derivatives,N_derivatives) .* (dt^j) .* view(uv_matrix, 1:system_size, 1+j)
    end
end

"""
Non-mutating version
"""
function arbitrary_RHS(uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = size(uv_matrix, 1)
    RHS = zeros(system_size)

    for j in 0:N_derivatives
        RHS += coefficient(j,N_derivatives, N_derivatives) * (dt^j) * view(uv_matrix, 1:system_size, 1+j)
    end
end

function arbitrary_LHS!(LHS::AbstractVector{Float64}, uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = length(LHS)
    @assert system_size == size(uv_matrix, 1)

    LHS .= 0.0
    for j in 0:N_derivatives
        LHS .+= (-1)^j .* coefficient(j,N_derivatives, N_derivatives) .* (dt^j) .* view(uv_matrix, 1:system_size, 1+j)
    end
end

"""
Non-mutating version
"""
function arbitrary_LHS(uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = size(uv_matrix, 1)
    LHS = zeros(system_size)

    for j in 0:N_derivatives
        LHS += (-1)^j * coefficient(j,N_derivatives, N_derivatives) * (dt^j) * view(uv_matrix, 1:system_size, 1+j)
    end
end
