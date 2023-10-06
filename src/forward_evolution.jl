"""
"""
function eval_forward(
        prob::SchrodingerProb, control::Control{Nderivatives},
        pcof::AbstractVector{Float64}; order=2, return_time_derivatives=false
    ) where Nderivatives

    if div(order, 2) > Nderivatives
        @warn "Calling method of order $order for control with $Nderivatives given. Obtaining higher order derivatives using automatic differentiation.\n"
    end

    if order == 2
        return eval_forward_order2(prob, control, pcof, return_time_derivatives=return_time_derivatives)
    elseif order == 4
        return eval_forward_order4(prob, control, pcof, return_time_derivatives=return_time_derivatives)
    end
    throw("Invalid order: $order")
end



"""
Evolve a single initial condition (vector).
"""
function eval_forward_order2(
        prob::SchrodingerProb{M, V}, control::Control{Nderivatives},
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, Nderivatives}
    
    @assert Nderivatives >= 1

    t = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(2*prob.N_tot_levels)
    copyto!(uv,1, prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)

    uv_history = Matrix{Float64}(undef,   2*prob.N_tot_levels, 1+prob.nsteps)
    uv_history[:,1] .= uv
    utvt_history = Matrix{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps)

    RHSu::Vector{Float64} = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64} = zeros(prob.N_tot_levels)
    RHS::Vector{Float64} = zeros(2*prob.N_tot_levels)

    u = copy(prob.u0)
    v = copy(prob.v0)
    ut = zeros(prob.N_tot_levels)
    vt = zeros(prob.N_tot_levels)

    # Order 2
    for n in 0:prob.nsteps-1
        utvt!(
            ut, vt, u, v,
            prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
            control.p[1], control.q[1], t, pcof
        )

        utvt_history[1:prob.N_tot_levels,     1+n] .= ut
        utvt_history[1+prob.N_tot_levels:end, 1+n] .= vt

        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS, 1, RHSu, 1, prob.N_tot_levels)
        copyto!(RHS, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func(
                ut, vt, uv[1:prob.N_tot_levels], uv[1+prob.N_tot_levels:end],
                prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
                control.p[1], control.q[1], t, pcof, dt, prob.N_tot_levels
            ),
            2*prob.N_tot_levels,
            2*prob.N_tot_levels
        )

        gmres!(uv, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+n+1] .= uv
        u = uv[1:prob.N_tot_levels]
        v = uv[1+prob.N_tot_levels:end]
    end

    # One last time, for utvt history at final time
    utvt!(ut, vt, u, v,
          prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
          control.p[1], control.q[1], t, pcof)

    utvt_history[1:prob.N_tot_levels,1+prob.nsteps] .= ut
    utvt_history[1+prob.N_tot_levels:end,1+prob.nsteps] .= vt

    if return_time_derivatives
        return cat(uv_history, utvt_history, dims=3)
    end
    return uv_history
end


"""
Evolve a matrix where each column is an iniital condition.
"""
function eval_forward_order2(
        prob::SchrodingerProb{M1, M2}, control::Control{Nderivatives},
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, Nderivatives}

    uv_history = Array{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps, prob.N_ess_levels)
    uv_utvt_history = Array{Float64}(undef,2*prob.N_tot_levels,1+prob.nsteps, 2, prob.N_ess_levels)

    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:prob.N_ess_levels

        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)

        # Call vector version of forward evolution
        if return_time_derivatives
            uv_utvt_history[:,:,:,initial_condition_index] .= eval_forward_order2(
                vector_prob, control, pcof, return_time_derivatives=true
            )
        else
            uv_history[:,:,initial_condition_index] .= eval_forward_order2(
                vector_prob, control, pcof, return_time_derivatives=false
            )
        end
    end
    
    if return_time_derivatives
        return uv_utvt_history
    else
        return uv_history
    end
end



"""
Evolve a single initial condition (vector).
"""
function eval_forward_order4(
        prob::SchrodingerProb{M, V}, control::Control{Nderivatives},
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, Nderivatives}

    t = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(prob.N_tot_levels*2)
    copyto!(uv, 1,                   prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)

    uv_history = Matrix{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps)
    uv_history[:,1] .= uv

    utvt_history   = Matrix{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps)
    uttvtt_history = Matrix{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps)

    RHSu::Vector{Float64} = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64} = zeros(prob.N_tot_levels)
    RHS::Vector{Float64}  = zeros(2*prob.N_tot_levels)

    u = copy(prob.u0)
    v = copy(prob.v0)
    ut  = zeros(prob.N_tot_levels)
    vt  = zeros(prob.N_tot_levels)
    utt = zeros(prob.N_tot_levels)
    vtt = zeros(prob.N_tot_levels)

    # Order 4
    for n in 0:prob.nsteps-1
        utvt!(
            ut, vt, u, v, prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
            control.p[1], control.q[1], t, pcof
        )
        uttvtt!(
            utt, vtt, ut, vt, u, v, prob.Ks, prob.Ss, prob.p_operator,
            prob.q_operator, control.p[1], control.q[1], control.p[2],
            control.q[2], t, pcof
        )

        utvt_history[1:prob.N_tot_levels,       1+n] .= ut
        utvt_history[1+prob.N_tot_levels:end,   1+n] .= vt
        uttvtt_history[1:prob.N_tot_levels,     1+n] .= utt
        uttvtt_history[1+prob.N_tot_levels:end, 1+n] .= vtt

        weights = [1,1/3]
        copy!(RHSu,u)
        axpy!(0.5*dt*weights[1],    ut,  RHSu)
        axpy!(0.25*dt^2*weights[2], utt, RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt*weights[1],    vt,  RHSv)
        axpy!(0.25*dt^2*weights[2], vtt, RHSv)

        copyto!(RHS, 1,                   RHSu, 1, prob.N_tot_levels)
        copyto!(RHS, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func_order4(
                utt, vtt, ut, vt, 
                uv[1:prob.N_tot_levels], uv[1+prob.N_tot_levels:end],
                prob.Ks, prob.Ss, prob.p_operator, prob.q_operator, 
                control.p[1], control.q[1], control.p[2], control.q[2],
                t, pcof, dt, prob.N_tot_levels
            ),
            2*prob.N_tot_levels, 2*prob.N_tot_levels
        )

        gmres!(uv, LHS_map, RHS)
        uv_history[:,1+n+1] .= uv

        u = uv[1:prob.N_tot_levels]
        v = uv[1+prob.N_tot_levels:end]
    end

    # One last time, for utvt history at final time
    utvt!(
        ut, vt, u, v,
        prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
        control.p[1], control.q[1], t, pcof
    )
    uttvtt!(
        utt, vtt, ut, vt, u, v,
        prob.Ks, prob.Ss, prob.p_operator, prob.q_operator,
        control.p[1], control.q[1], control.p[2], control.q[2], t, pcof
    )

    utvt_history[1:prob.N_tot_levels,       1+prob.nsteps] .= ut
    utvt_history[1+prob.N_tot_levels:end,   1+prob.nsteps] .= vt
    uttvtt_history[1:prob.N_tot_levels,     1+prob.nsteps] .= utt
    uttvtt_history[1+prob.N_tot_levels:end, 1+prob.nsteps] .= vtt

    if return_time_derivatives
        return cat(uv_history, utvt_history, uttvtt_history, dims=3)
    end
    return uv_history
end

function eval_forward_order4(
        prob::SchrodingerProb{M1, M2}, control::Control{Nderivatives},
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, Nderivatives}

    N_init_cond = size(prob.u0,2)
    uv_history = Array{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps, N_init_cond)
    uv_and_derivatives_history = Array{Float64}(undef, 2*prob.N_tot_levels, 1+prob.nsteps, 3, N_init_cond)

    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:N_init_cond

        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)

        # Call vector version of forward evolution
        if return_time_derivatives
            uv_utvt_history[:,:,:,initial_condition_index] .= eval_forward_order4(
                vector_prob, control, pcof, return_time_derivatives=true
            )
        else
            uv_history[:,:,initial_condition_index] .= eval_forward_order4(
                vector_prob, control, pcof, return_time_derivatives=false
            )
        end
    end
    
    if return_time_derivatives
        return uv_utvt_history
    else
        return uv_history
    end
end

function eval_forward_forced(
        prob::SchrodingerProb{M, V}, forcing_ary,
        pcof::AbstractVector{Float64}; order=2
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
    if order == 2
        return eval_forward_forced_order2(prob, forcing_ary, pcof)
    elseif order == 4
        return eval_forward_forced_order4(prob, forcing_ary, pcof)
    end

    throw("Invalid Order: $order")
end

"""
Evolve schrodinger problem with forcing applied, and forcing given as an array
of forces at each discretized point in time.

Maybe I should also do a one with forcing functions as well.
"""
function eval_forward_forced_order2(
        prob::SchrodingerProb{M, V}, forcing_ary::AbstractArray{Float64,3}, pcof::V
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.p_operator
    a_minus_adag = prob.q_operator
    p = prob.p
    q = prob.q
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps
    N_ess = prob.N_ess_levels
    N_grd = prob.N_guard_levels
    N_tot = prob.N_tot_levels

    t = 0.0
    dt = tf/nsteps

    uv = zeros(2*N_tot)
    copyto!(uv,1,u0,1,N_tot)
    copyto!(uv,1+N_tot,v0,1,N_tot)
    uv_history = Matrix{Float64}(undef,2*N_tot,1+nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64} = zeros(N_tot)
    RHSv::Vector{Float64} = zeros(N_tot)
    RHS_uv::Vector{Float64} = zeros(2*N_tot)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(N_tot)
    vt = zeros(N_tot)
    utt = zeros(N_tot)
    vtt = zeros(N_tot)

    for n in 0:nsteps-1
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, pcof)
        copy!(RHSu,u)
        axpy!(0.5*dt,ut,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt,vt,RHSv)

        copyto!(RHS_uv,1,RHSu,1,N_tot)
        copyto!(RHS_uv,1+N_tot,RHSv,1,N_tot)

        axpy!(0.5*dt,forcing_ary[:,1+n,1], RHS_uv)
        axpy!(0.5*dt,forcing_ary[:,1+n+1,1], RHS_uv)

        t += dt

        LHS_map = LinearMap(
            uv -> LHS_func(ut, vt, uv[1:N_tot], uv[1+N_tot:end],
                           Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, t, pcof, dt, N_tot),
            2*N_tot,2*N_tot
        )

        gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+n+1] .= uv
        u = uv[1:N_tot]
        v = uv[1+N_tot:end]
    end
    
    return uv_history
end



function eval_forward_forced_order4(
        prob::SchrodingerProb{M, V}, forcing_ary::AbstractArray{Float64,3}, pcof::V
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
    Ks = prob.Ks
    Ss = prob.Ss
    a_plus_adag = prob.p_operator
    a_minus_adag = prob.q_operator
    p = prob.p
    q = prob.q
    dpdt = prob.dpdt
    dqdt = prob.dqdt
    u0 = prob.u0
    v0 = prob.v0
    tf = prob.tf
    nsteps = prob.nsteps
    N_ess = prob.N_ess_levels
    N_grd = prob.N_guard_levels
    N_tot = prob.N_tot_levels

    t = 0.0
    dt = tf/nsteps

    uv = zeros(2*N_tot)
    copyto!(uv,1,u0,1,N_tot)
    copyto!(uv,1+N_tot,v0,1,N_tot)
    uv_history = Matrix{Float64}(undef,2*N_tot,1+nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64} = zeros(N_tot)
    RHSv::Vector{Float64} = zeros(N_tot)
    RHS_uv::Vector{Float64} = zeros(2*N_tot)

    u = copy(u0)
    v = copy(v0)
    ut = zeros(N_tot)
    vt = zeros(N_tot)
    utt = zeros(N_tot)
    vtt = zeros(N_tot)

    weights = [1,1/3]
    weights_LHS = [1,-1/3]
    for n in 0:nsteps-1
        # First time derivative at current timestep
        utvt!(ut, vt, u, v,
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, pcof)
        axpy!(1.0,forcing_ary[1:N_tot,1+n,1], ut)
        axpy!(1.0,forcing_ary[1+N_tot:end,1+n,1], vt)

        # Second time derivative at current timestep
        uttvtt!(utt, vtt, ut, vt, u, v,
                Ks, Ss, a_plus_adag, a_minus_adag,
                p, q, dpdt, dqdt, t, pcof)
        axpy!(1.0,forcing_ary[1:N_tot,1+n,2], utt)
        axpy!(1.0,forcing_ary[1+N_tot:end,1+n,2], vtt)

        # Accumulate RHS (current time contributions)
        copy!(RHSu,u)
        axpy!(0.5*dt*weights[1],ut,RHSu)
        axpy!(0.25*dt^2*weights[2],utt,RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt*weights[1],vt,RHSv)
        axpy!(0.25*dt^2*weights[2],vtt,RHSv)

        # Add in forcing from first time derivative at next timestep
        axpy!(0.5*dt*weights_LHS[1],forcing_ary[1:N_tot,1+n+1,1], RHSu)
        axpy!(0.5*dt*weights_LHS[1],forcing_ary[1+N_tot:end,1+n+1,1], RHSv)

        # Add in forcing from second time derivative at next timestep
        axpy!(0.25*dt^2*weights_LHS[2],forcing_ary[1:N_tot,1+n+1,2], RHSu)
        axpy!(0.25*dt^2*weights_LHS[2],forcing_ary[1+N_tot:end,1+n+1,2], RHSv)

        # Advance time to next point
        t += dt

        # Don't forget to differentiate the forcing from the first derivative
        utvt!(ut, vt, forcing_ary[1:N_tot,1+n+1,1], forcing_ary[1+N_tot:end,1+n+1,1],
              Ks, Ss, a_plus_adag, a_minus_adag,
              p, q, t, pcof)

        axpy!(0.25*dt^2*weights_LHS[2], ut, RHSu)
        axpy!(0.25*dt^2*weights_LHS[2], vt, RHSv)

        copyto!(RHS_uv,1,RHSu,1,N_tot)
        copyto!(RHS_uv,1+N_tot,RHSv,1,N_tot)

        LHS_map = LinearMap(
            uv -> LHS_func_order4(utt, vtt, ut, vt, uv[1:N_tot], uv[1+N_tot:end],
                           Ks, Ss, a_plus_adag, a_minus_adag,
                           p, q, dpdt, dqdt, t, pcof, dt, N_tot),
            2*N_tot,2*N_tot
        )

        gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+n+1] .= uv
        u = uv[1:N_tot]
        v = uv[1+N_tot:end]
    end
    
    return uv_history
end
