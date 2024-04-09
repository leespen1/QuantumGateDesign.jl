"""
Working on arbitrary order version of utvt!, uttvtt!, etc.

uv_matrix's first column is u and v stacked. Second column is ut and vt stacked, etc.

u and v should be given, and the derivatives are to be computed in place by
this method.

"""
function compute_derivatives!(uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64; use_adjoint::Bool=false,
        forcing_matrix::Union{AbstractMatrix{Float64}, Missing}=missing
    )

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end

    # Calculate each derivative
    for j = 0:(N_derivatives-1)
        # Get views of the current derivative we are trying to compute (the j+1th derivative)
        u_derivative = view(uv_matrix, 1:prob.N_tot_levels,                       1+j+1)
        v_derivative = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j+1)

        u_derivative .= 0
        v_derivative .= 0

        # Perform the summation
        for i = j:-1:0
            derivative_order = j-i

            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            apply_hamiltonian!(u_derivative, v_derivative, u_derivative_prev,
                v_derivative_prev, prob, controls, t, pcof,
                derivative_order=derivative_order
            )
        end

        # Apply forcing
        if !ismissing(forcing_matrix)
            axpy!(1.0, view(forcing_matrix, 1:prob.N_tot_levels,                       1+j), u_derivative)
            axpy!(1.0, view(forcing_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j), v_derivative)
        end

        # Complete the 1/(j+1)! factor
        u_derivative ./= j+1
        v_derivative ./= j+1
    end
    
    return nothing
end

"""
lambda_in = Λ₀

Returns Λ_(derivative_index)

Need to describe the recursion better.
"""
function compute_single_adjoint_derivative(lambda_in::AbstractVector{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        derivative_index::Int64; 
    )

    if (derivative_index == 0)
        return lambda_in
    end

    lambda_out = zeros(length(lambda_in))

    lambda_in_u = view(lambda_in, 1:prob.N_tot_levels)
    lambda_in_v = view(lambda_in, 1+prob.N_tot_levels:prob.real_system_size)

    lambda_in_temp = copy(lambda_in)
    lambda_in_temp_u = view(lambda_in_temp, 1:prob.N_tot_levels)
    lambda_in_temp_v = view(lambda_in_temp, 1+prob.N_tot_levels:prob.real_system_size)

    # Iterate over derivative orders lower than the one we are trying to calculate
    for derivative_order=(derivative_index-1):-1:0

        lambda_in_temp_u .= 0
        lambda_in_temp_v .= 0
        
        # Apply A_(derivative_i) to Λ₀, to get our "new Λ₀"
        apply_hamiltonian!(lambda_in_temp_u, lambda_in_temp_v, 
            lambda_in_u, lambda_in_v,
            prob, controls, t, pcof,
            derivative_order=derivative_order,
            use_adjoint=true
        )
        
        lambda_out .+= compute_single_adjoint_derivative(
            lambda_in_temp, prob, controls, t, pcof, (derivative_index-1)-derivative_order
        )
    end

    return lambda_out ./ derivative_index
end

"""
WIP : Fixing the math, canrt just add an adjoint factor into uv_derivative

I don't think the input should have 0 in the first column.
"""
function compute_adjoint_derivatives!(
        uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64; 
    )

    lambda_in = uv_matrix[:,1]
    for derivative_i in 1:N_derivatives
        uv_matrix[:,1+derivative_i] .= compute_single_adjoint_derivative(
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

For efficiency when using automatic differentiation, it could be useful to just
give all the values of the control gradients in one array.
"""
function compute_partial_derivative!(
        uv_partial_matrix::AbstractMatrix{Float64}, uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64, global_pcof_index::Int64
    )

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end

    uv_partial_matrix[:,1] .= 0 # ∂w/∂θₖ = 0

    # Calculate ∂/∂θₖ for each time derivative
    for j = 0:(N_derivatives-1)
        # Get views of the current derivative we are trying to compute (the j+1th derivative)
        u_partial_derivative = view(uv_partial_matrix, 1:prob.N_tot_levels,                       1+j+1)
        v_partial_derivative = view(uv_partial_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j+1)

        u_partial_derivative .= 0
        v_partial_derivative .= 0

        # Perform the summation
        for i = j:-1:0
            derivative_order = (j-i)

            # A(∂w/∂θₖ) part
            u_partial_derivative_prev = view(uv_partial_matrix, 1:prob.N_tot_levels,                       1+i)
            v_partial_derivative_prev = view(uv_partial_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            apply_hamiltonian!(u_partial_derivative, v_partial_derivative, 
                u_partial_derivative_prev, v_partial_derivative_prev,
                prob, controls, t, pcof,
                derivative_order=derivative_order
            )


            # (∂A/∂θₖ)w part (only involves one control)
            u_derivative_prev = view(uv_matrix, 1:prob.N_tot_levels,                       1+i)
            v_derivative_prev = view(uv_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+i)

            control_index, local_index = get_local_control_indices(controls, global_pcof_index)
            
            local_control = controls[control_index]
            local_pcof = get_control_vector_slice(pcof, controls, control_index)
            sym_op = prob.sym_operators[control_index]
            asym_op = prob.asym_operators[control_index]

            #p_val = eval_grad_p_derivative(local_control, t, local_pcof, derivative_order)[local_index] / factorial(derivative_order)
            #q_val = eval_grad_q_derivative(local_control, t, local_pcof, derivative_order)[local_index] / factorial(derivative_order)
            ## Changing to non-allocating version, handle on a per-control-parameter basis
            p_val = eval_grad_p_derivative(local_control, t, local_pcof, derivative_order, local_index) / factorial(derivative_order)
            q_val = eval_grad_q_derivative(local_control, t, local_pcof, derivative_order, local_index) / factorial(derivative_order)

            mul!(u_partial_derivative, asym_op, u_derivative_prev, q_val, 1)
            mul!(u_partial_derivative, sym_op,  v_derivative_prev, p_val, 1)

            mul!(v_partial_derivative, asym_op, v_derivative_prev, q_val,  1)
            mul!(v_partial_derivative, sym_op,  u_derivative_prev, -p_val, 1)
        end

        u_partial_derivative ./= j+1
        v_partial_derivative ./= j+1
    end
    
    return nothing
end


function coefficient(j,p,q)
    return factorial(p)*factorial(p+q-j)/(factorial(p+q)*factorial(p-j))
end


function build_RHS!(RHS::AbstractVector{Float64}, uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)
    RHS .= 0.0

    for j in 0:N_derivatives
        coeff = dt^j * coefficient(j, N_derivatives, N_derivatives)
        derivative_vec = view(uv_matrix, :, 1+j)
        LinearAlgebra.axpy!(coeff, derivative_vec, RHS)
    end
end

"""
Non-mutating version
"""
function build_RHS(uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = size(uv_matrix, 1)
    RHS = zeros(system_size)

    build_RHS!(RHS, uv_matrix, dt, N_derivatives)
end


function build_LHS!(LHS::AbstractVector{Float64}, uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)
    LHS .= 0.0

    for j in 0:N_derivatives
        coeff = (-dt)^j * coefficient(j, N_derivatives, N_derivatives)
        derivative_vec = view(uv_matrix, :, 1+j)
        LinearAlgebra.axpy!(coeff, derivative_vec, LHS)
    end
end

"""
Non-mutating version
"""
function build_LHS(uv_matrix::AbstractMatrix{Float64},
        dt::Real, N_derivatives::Int64)

    system_size = size(uv_matrix, 1)
    LHS = zeros(system_size)

    build_LHS!(LHS, uv_matrix, dt, N_derivatives)
end

"""
Given a matrix whose columns are a vector and its derivatives at time t,
perform taylor expansion to approximate the vector at time t+dt.

Overwrite out_vec with the result.
"""
function taylor_expand!(out_vec::AbstractVector{Float64}, uv_matrix::AbstractMatrix{Float64}, dt::Float64, N_derivatives::Int64)
    out_vec .= 0.0

    for j in 0:N_derivatives
        taylor_coeff = dt^j / factorial(j)
        derivative_vec = view(uv_matrix, :, 1+j)
        LinearAlgebra.axpy!(taylor_coeff, derivative_vec, out_vec)
    end

    return nothing
end

"""
Apply hamiltonian to 'in', *add* result to out.

For derivatives, apply Hamiltonian divided by the factorial of the derivative order.
"""
function apply_hamiltonian!(out_real::AbstractVector{Float64}, out_imag::AbstractVector{Float64},
        in_real::AbstractVector{Float64}, in_imag::AbstractVector{Float64},
        prob::SchrodingerProb, controls, t::Float64,
        pcof::AbstractVector{<: Real}; derivative_order::Int=0, use_adjoint::Bool=false
    )

    adjoint_factor = use_adjoint ? -1 : 1

    # Apply system hamiltonian if we aren't taking the derivative of the hamiltonian
    if (derivative_order == 0)
        # Each of these `mul!`s does a single allocation of 64 bytes. I think
        # that accounts for the allocations, and is consistent with REPL
        # (actually less). So I need to fix this
        #
        # When I just do a loop with a bunch of apply_hamiltonian! calls, the
        # allocation doesn't happen. So it appears to be a type inference problem.
        # Try Cthulhu
        #
        # I guess it's not type-instability, but mul! allocates memory when the matrix is sparse and the vectors are views. 
        # So what I guess I need are some working arrays.
        mul!(out_real, prob.system_asym, in_real, adjoint_factor, 1)
        mul!(out_real, prob.system_sym,  in_imag, adjoint_factor, 1)

        mul!(out_imag, prob.system_asym, in_imag, adjoint_factor, 1)
        mul!(out_imag, prob.system_sym,  in_real, -adjoint_factor, 1)
    end

    # Apply the control hamiltonian(s)
    for k in 1:prob.N_operators
        local_control = controls[k]
        sym_op = prob.sym_operators[k]
        asym_op = prob.asym_operators[k]
        # Get the part of the full control vector which corresponds to this particular control
        local_pcof = get_control_vector_slice(pcof, controls, k)

        p_val = eval_p_derivative(local_control, t, local_pcof, derivative_order) / factorial(derivative_order)
        q_val = eval_q_derivative(local_control, t, local_pcof, derivative_order) / factorial(derivative_order)

        mul!(out_real, asym_op, in_real, adjoint_factor*q_val, 1)
        mul!(out_real, sym_op,  in_imag, adjoint_factor*p_val, 1)

        mul!(out_imag, asym_op, in_imag, adjoint_factor*q_val,  1)
        mul!(out_imag, sym_op,  in_real, -adjoint_factor*p_val, 1)
    end
end

"""
Construct the LHS matrix from the timestep. Not used in the actual algorithm,
but good for checking condition numbers.
"""
function form_LHS(prob::SchrodingerProb, controls, t::Real,
        pcof::AbstractVector{<: Real}, dt::Real, order::Int)
    
    real_system_size = prob.real_system_size
    complex_system_size = div(real_system_size, 2)
    N_derivatives = div(order, 2)

    LHS_mat = zeros(real_system_size, real_system_size)
    LHS_vec = zeros(real_system_size)
    uv_matrix = zeros(real_system_size, 1+N_derivatives)
    for i in 1:real_system_size
        uv_matrix .= 0
        LHS_vec .= 0
        uv_matrix[i,1] = 1
        compute_derivatives!(uv_matrix, prob, controls, t, pcof, N_derivatives)
        build_LHS!(LHS_vec, uv_matrix, dt, N_derivatives)
        LHS_mat[:,i] .= LHS_vec
    end

    return LHS_mat
end

"""
Construct the RHS matrix (but don't multiply onto vector) from the timestep. Not used in the actual algorithm,
but good for checking condition numbers.
"""
function form_RHS(prob::SchrodingerProb, controls, t::Real,
        pcof::AbstractVector{<: Real}, dt::Real, order::Int)
    
    real_system_size = prob.real_system_size
    complex_system_size = div(real_system_size, 2)
    N_derivatives = div(order, 2)

    RHS_mat = zeros(real_system_size, real_system_size)
    RHS_vec = zeros(real_system_size)
    uv_matrix = zeros(real_system_size, 1+N_derivatives)
    for i in 1:real_system_size
        uv_matrix .= 0
        RHS_vec .= 0
        uv_matrix[i,1] = 1
        compute_derivatives!(uv_matrix, prob, controls, t, pcof, N_derivatives)
        build_RHS!(RHS_vec, uv_matrix, dt, N_derivatives)
        RHS_mat[:,i] .= RHS_vec
    end

    return RHS_mat
end

"""
Construct the LHS and RHS matrices from the timestep, with appropraite times
from each.
"""
function form_LHS_RHS(prob::SchrodingerProb, controls, t::Real,
        pcof::AbstractVector{<: Real}, dt::Real, order::Int)
    tn = t
    tnp1 = t + dt

    RHS = form_RHS(prob, controls, tn, pcof, dt, order)
    LHS = form_LHS(prob, controls, tnp1, pcof, dt, order)

    return LHS, RHS
end

