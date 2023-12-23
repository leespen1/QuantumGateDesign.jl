#==============================================================================
#
# I should take a look at this again and give things more informative names
#
==============================================================================#
"""
Values of p(t,pcof) and q(t,pcof) provided. Mutates ut and vt, leaves all other variables untouched.

Could do this with fewer matrix multiplications by having a storage matrix
where I store system_sym + p(t)*sym_op, and the corresponding asym part. This
would reduce the number of matrix multiplications by half, at the cost of
matrix additions. For N controls, it would reduce the number of matrix
multiplications by 1/N.
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64},
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64)
    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u + (Ks + p(t)(a+a†))v
    mul!(ut, system_asym, u)
    mul!(ut, control_op_asym, u, q_val, 1)
    mul!(ut, system_sym, v, 1, 1)
    mul!(ut, control_op_sym, v, p_val, 1)

    # vt = (Ss + q(t)(a-a†))v - (Ks + p(t)(a+a†))u
    mul!(vt, system_asym, v)
    mul!(vt, control_op_asym, v, q_val, 1)
    mul!(vt, system_sym, u, -1, 1)
    mul!(vt, control_op_sym,  u, -p_val, 1)

    return nothing
end

"""
Multiple control version
"""
function utvt!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real})

    # Non-Memory-Allocating Version (test performance)
    # ut = (Ss + q(t)(a-a†))u + (Ks + p(t)(a+a†))v
    # vt = (Ss + q(t)(a-a†))v - (Ks + p(t)(a+a†))u
    
    mul!(ut, prob.system_asym, u)
    mul!(ut, prob.system_sym, v, 1, 1)

    mul!(vt, prob.system_asym, v)
    mul!(vt, prob.system_sym, u, -1, 1)

    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(ut, asym_op, u, eval_q(control, t, this_pcof), 1)
        mul!(ut, sym_op, v, eval_p(control, t, this_pcof), 1)

        mul!(vt, asym_op, v, eval_q(control, t, this_pcof), 1)
        mul!(vt, sym_op,  u, -eval_p(control, t, this_pcof), 1)
    end

    return nothing
end

"""
Evaluate first derivative in adjoint equation.

In the real-valued formulation `y = Ax`, the adjoint equation is `y = Aᵀx`.
Because the Hamiltonian is Hermitian, for the adjoint equation we only need to
do 
[S K; -K S]ᵀ = [-S -K; K -S]
"""
function utvt_adj!(ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real})

    # System / Drift Part of Hamiltonian
    mul!(ut, prob.system_asym, u)
    ut .*= -1 # Want to do ut=Sᵀu = -Su, this is the best way to get the negative without allocating memory
    mul!(ut, prob.system_sym, v, -1, 1)

    mul!(vt, prob.system_asym, v)
    vt .*= -1 # Get negative without allocating memory
    mul!(vt, prob.system_sym, u, 1, 1)

    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(ut, asym_op, u, -eval_q(control, t, this_pcof), 1)
        mul!(ut, sym_op, v, -eval_p(control, t, this_pcof), 1)

        mul!(vt, sym_op,  u, eval_p(control, t, this_pcof), 1)
        mul!(vt, asym_op, v, -eval_q(control, t, this_pcof), 1)
    end

    return nothing
end


"""
Assumes that ut and vt have already been computed

Overwrites utt and vtt, leaves everything else untouched.

ψtt = Ht*ψ + H*ψt = Ht*ψ + H²*ψ

Hψ is already calculated and stored in ut/vt. Therefore the call to utvt! with ut/vt in
place of u/v computes H²ψ.

The rest of the function computes Ht*ψb

"""
function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64},
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64,
        dpdt_val::Float64, dqdt_val::Float64
        )
    ## Make use of utvt! to compute H²ψ, make use of ψt already computed
    utvt!(utt, vtt, ut, vt, system_sym, system_asym, control_op_sym, control_op_asym, p_val, q_val)

    # Add Ht*ψ. System/drift hamiltonian is time-independent, falls out in Ht
    mul!(utt, control_op_asym, u, dqdt_val, 1)
    mul!(utt, control_op_sym,  v, dpdt_val, 1)

    mul!(vtt, control_op_sym,  u, -dpdt_val, 1)
    mul!(vtt, control_op_asym, v, dqdt_val,  1)

    return nothing
end

function uttvtt!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real})

    # Calculate H²ψ
    utvt!(utt, vtt, ut, vt, prob, controls, t, pcof)

    # Calculate Hₜψ (time derivative means drift hamiltonian goes away)
    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(utt, asym_op, u, eval_qt(control, t, this_pcof), 1)
        mul!(utt, sym_op,  v, eval_pt(control, t, this_pcof), 1)

        mul!(vtt, sym_op,  u, -eval_pt(control, t, this_pcof), 1)
        mul!(vtt, asym_op, v, eval_qt(control, t, this_pcof),  1)
    end

end

function utttvttt!(
        uttt::AbstractVector{Float64}, vttt::AbstractVector{Float64},
        utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real})

    utvt!(uttt, vttt, utt, vtt, prob, controls, t, pcof)

    # 2Aₜyₜ
    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(uttt, asym_op, ut, 2*eval_qt(control, t, this_pcof), 1)
        mul!(uttt, sym_op,  vt, 2*eval_pt(control, t, this_pcof), 1)

        mul!(vttt, sym_op,  ut, -2*eval_pt(control, t, this_pcof), 1)
        mul!(vttt, asym_op, vt, 2*eval_qt(control, t, this_pcof),  1)
    end

    # Aₜₜy
    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(uttt, asym_op, u, eval_q_derivative(control, t, this_pcof, 2), 1)
        mul!(uttt, sym_op,  v, eval_p_derivative(control, t, this_pcof, 2), 1)

        mul!(vttt, sym_op,  u, -eval_p_derivative(control, t, this_pcof, 2), 1)
        mul!(vttt, asym_op, v, eval_q_derivative(control, t, this_pcof, 2),  1)
    end

end

function uttvtt_adj!(utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real})

    # Calculate H²ψ
    utvt_adj!(utt, vtt, ut, vt, prob, controls, t, pcof)

    # Calculate Hₜψ (time derivative means drift hamiltonian goes away)
    for i in 1:prob.N_operators
        control = controls[i]
        sym_op = prob.sym_operators[i]
        asym_op = prob.asym_operators[i]

        # Get part of control vector corresponding to this control
        this_pcof = get_control_vector_slice(pcof, controls, i)

        mul!(utt, asym_op, u, -eval_qt(control, t, this_pcof), 1)
        mul!(utt, sym_op,  v, -eval_pt(control, t, this_pcof), 1)

        mul!(vtt, sym_op,  u, eval_pt(control,  t, this_pcof), 1)
        mul!(vtt, asym_op, v, -eval_qt(control, t, this_pcof), 1)
    end

end


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
For use in discrete adjoint (maybe it will also be useful in forced gradient)

uv_matrix should already be filled.

The first column of uv_partial_matrix should be zeros.

Maybe forcing matrix should be named forcing_partial_matrix. Because although
it functions the same as before, it is a different thing mathematically.


"""
function arbitrary_order_uv_partial_derivative!(
        uv_partial_matrix::AbstractMatrix{Float64}, uv_matrix::AbstractMatrix{Float64},
        prob::SchrodingerProb, controls, t::Float64, pcof::AbstractVector{<: Real},
        N_derivatives::Int64, global_pcof_index::Int64; use_adjoint::Bool=false,
        forcing_matrix::Union{AbstractMatrix{Float64}, Missing}=missing
    )

    if (N_derivatives < 1)
        throw(ArgumentError("Non positive N_derivatives supplied."))
    end

    adjoint_factor = use_adjoint ? -1 : 1

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
        mul!(u_partial_derivative, prob.system_asym, u_partial_derivative_prev, adjoint_factor, 1)
        mul!(u_partial_derivative, prob.system_sym,  v_partial_derivative_prev, adjoint_factor, 1)

        mul!(v_partial_derivative, prob.system_asym, v_partial_derivative_prev, adjoint_factor, 1)
        mul!(v_partial_derivative, prob.system_sym,  u_partial_derivative_prev, -adjoint_factor, 1)


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

                mul!(u_partial_derivative, asym_op, u_partial_derivative_prev, adjoint_factor*q_val, 1)
                mul!(u_partial_derivative, sym_op,  v_partial_derivative_prev, adjoint_factor*p_val, 1)

                mul!(v_partial_derivative, asym_op, v_partial_derivative_prev, adjoint_factor*q_val,  1)
                mul!(v_partial_derivative, sym_op,  u_partial_derivative_prev, -adjoint_factor*p_val, 1)
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

            mul!(u_partial_derivative, asym_op, u_derivative_prev, adjoint_factor*q_val, 1)
            mul!(u_partial_derivative, sym_op,  v_derivative_prev, adjoint_factor*p_val, 1)

            mul!(v_partial_derivative, asym_op, v_derivative_prev, adjoint_factor*q_val,  1)
            mul!(v_partial_derivative, sym_op,  u_derivative_prev, -adjoint_factor*p_val, 1)
        end

        # I believe checking like this means that if-block will be compiled out when no forcing matrix is given
        if !ismissing(forcing_matrix)
            axpy!(1.0, view(forcing_matrix, 1:prob.N_tot_levels,                       1+j), u_derivative)
            axpy!(1.0, view(forcing_matrix, prob.N_tot_levels+1:prob.real_system_size, 1+j), v_derivative)
        end

        # Pretty sure I do want the 1/j+1 in this one
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

function LHS_func!(LHS_uv::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64},
        control_op_sym::AbstractMatrix{Float64}, control_op_asym::AbstractMatrix{Float64},
        p_val::Float64, q_val::Float64, dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v, system_sym, system_asym, control_op_sym, control_op_asym, p_val, q_val)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt, ut, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt, vt, LHS_v)

    return nothing
end

"""
In the Hermite timestepping method, we arrive at an equation like

LHS*uvⁿ⁺¹ = RHS*uvⁿ

This function computes the action of LHS on a vector uv.
That is, given an input uv (as two arguments u and v), return LHS*uv
"""
function LHS_func!(LHS_uv::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v, prob, controls, t, pcof)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt, ut, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt, vt, LHS_v)

    return nothing
end

function LHS_func_adj!(LHS_uv::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real},
        dt::Float64, N_tot::Int64)

    utvt_adj!(ut, vt, u, v, prob, controls, t, pcof)
    
    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)

    LHS_u .= u
    axpy!(-0.5*dt, ut, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt, vt, LHS_v)

    return nothing
end


function LHS_func_order4!(LHS_uv::AbstractVector{Float64},
        utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real},
        dt::Float64, N_tot::Int64)

    utvt!(ut, vt, u, v, prob, controls, t, pcof)
    uttvtt!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

    weights = (1,-1/3)

    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)

    LHS_u .= u
    axpy!(-0.5*dt*weights[1],     ut,  LHS_u)
    axpy!(-0.25*dt*dt*weights[2], utt, LHS_u)

    LHS_v .= v
    axpy!(-0.5*dt*weights[1],     vt,  LHS_v)
    axpy!(-0.25*dt*dt*weights[2], vtt, LHS_v)

    return nothing
end

function LHS_func_order4_adj!(LHS_uv::AbstractVector{Float64},
        utt::AbstractVector{Float64}, vtt::AbstractVector{Float64},
        ut::AbstractVector{Float64}, vt::AbstractVector{Float64},
        u::AbstractVector{Float64}, v::AbstractVector{Float64},
        prob::SchrodingerProb, controls,
        t::Float64, pcof::AbstractVector{<: Real},
        dt::Float64, N_tot::Int64)

    utvt_adj!(ut, vt, u, v, prob, controls, t, pcof)
    uttvtt_adj!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

    weights = (1,-1/3)
    LHS_u = view(LHS_uv, 1:N_tot)
    LHS_v = view(LHS_uv, 1+N_tot:2*N_tot)
    
    LHS_u .= u
    axpy!(-0.5*dt*weights[1],     ut,  LHS_u)
    axpy!(-0.25*dt*dt*weights[2], utt, LHS_u)
    LHS_v .= v
    axpy!(-0.5*dt*weights[1],     vt,  LHS_v)
    axpy!(-0.25*dt*dt*weights[2], vtt, LHS_v)

    return nothing
end
