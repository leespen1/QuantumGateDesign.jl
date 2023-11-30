"""
"""
function eval_forward(
        prob::SchrodingerProb, controls,
        pcof::AbstractVector{Float64}; order=2, return_time_derivatives=false
    )

    #=
    if div(order, 2) > Nderivatives
        @warn "Calling method of order $order for control with $Nderivatives given. Obtaining higher order derivatives using automatic differentiation.\n"
    end
    =#

    if order == 2
        return eval_forward_order2(prob, controls, pcof, return_time_derivatives=return_time_derivatives)
    elseif order == 4
        return eval_forward_order4(prob, controls, pcof, return_time_derivatives=return_time_derivatives)
    end
    throw("Invalid order: $order")
end



"""
Evolve a single initial condition (vector).
"""
function eval_forward_order2(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
    
    t::Float64 = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(prob.real_system_size)
    copyto!(uv, 1,                   prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)

    uv_history = Matrix{Float64}(undef,   prob.real_system_size, 1+prob.nsteps)
    uv_history[:,1] .= uv
    utvt_history = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)

    RHSu::Vector{Float64} = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64} = zeros(prob.N_tot_levels)
    RHS::Vector{Float64} = zeros(prob.real_system_size)

    u = zeros(prob.N_tot_levels)
    v = zeros(prob.N_tot_levels)

    u .= prob.u0
    v .= prob.v0

    ut = zeros(prob.N_tot_levels)
    vt = zeros(prob.N_tot_levels)


    # This probably creates type-instability, as functions have singleton types, 
    # and this function is (I'm pretty sure) not created until runtime
    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        # Careful, make certain that I can do this overwrite without messing up anything else
        copyto!(u, 1, uv_in, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv_in, 1+prob.N_tot_levels, prob.N_tot_levels)
        LHS_func!(uv_out, ut, vt, u, v, prob, controls, t, pcof, dt, prob.N_tot_levels)
        return nothing
    end

    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )


    # Order 2
    for n in 0:prob.nsteps-1
        utvt!(ut, vt, u, v, prob, controls, t, pcof)

        utvt_history[1:prob.N_tot_levels,     1+n] .= ut
        utvt_history[1+prob.N_tot_levels:end, 1+n] .= vt

        copy!(RHSu, u)
        axpy!(0.5*dt, ut, RHSu)

        copy!(RHSv, v)
        axpy!(0.5*dt, vt, RHSv)

        copyto!(RHS, 1,                   RHSu, 1, prob.N_tot_levels)
        copyto!(RHS, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        t += dt

        IterativeSolvers.gmres!(uv, LHS_map, RHS, abstol=1e-15, reltol=1e-15)

        uv_history[:,1+n+1] .= uv
        copyto!(u, 1, uv, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv, 1+prob.N_tot_levels, prob.N_tot_levels)

    end

    # One last time, for utvt history at final time
    utvt!(ut, vt, u, v, prob, controls, t, pcof)

    utvt_history[1:prob.N_tot_levels,     1+prob.nsteps] .= ut
    utvt_history[1+prob.N_tot_levels:end, 1+prob.nsteps] .= vt

    if return_time_derivatives
        return cat(uv_history, utvt_history, dims=3)
    end
    return uv_history
end


"""
Evolve a matrix where each column is an iniital condition.
"""
function eval_forward_order2(
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    uv_history = Array{Float64}(undef, prob.real_system_size, 1+prob.nsteps, prob.N_ess_levels)
    uv_utvt_history = Array{Float64}(undef,prob.real_system_size,1+prob.nsteps, 2, prob.N_ess_levels)

    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:prob.N_ess_levels

        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)

        # Call vector version of forward evolution
        if return_time_derivatives
            uv_utvt_history[:,:,:,initial_condition_index] .= eval_forward_order2(
                vector_prob, controls, pcof, return_time_derivatives=true
            )
        else
            uv_history[:,:,initial_condition_index] .= eval_forward_order2(
                vector_prob, controls, pcof, return_time_derivatives=false
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
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    t = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(prob.N_tot_levels*2)
    copyto!(uv, 1,                   prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)

    uv_history = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)
    uv_history[:,1] .= uv

    utvt_history   = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)
    uttvtt_history = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)

    RHSu::Vector{Float64} = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64} = zeros(prob.N_tot_levels)
    RHS::Vector{Float64}  = zeros(prob.real_system_size)

    u = copy(prob.u0)
    v = copy(prob.v0)
    ut  = zeros(prob.N_tot_levels)
    vt  = zeros(prob.N_tot_levels)
    utt = zeros(prob.N_tot_levels)
    vtt = zeros(prob.N_tot_levels)

    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        # Careful, make certain that I can do this overwrite without messing up anything else
        copyto!(u, 1, uv_in, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv_in, 1+prob.N_tot_levels, prob.N_tot_levels)
        LHS_func_order4!(
            uv_out, utt, vtt, ut, vt, u, v, prob, controls, t, pcof, dt, prob.N_tot_levels
        )
        return nothing
    end

    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    # Order 4
    weights = (1, 1/3)
    for n in 0:prob.nsteps-1
        utvt!(ut, vt, u, v, prob, controls, t, pcof)
        uttvtt!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

        utvt_history[1:prob.N_tot_levels,       1+n] .= ut
        utvt_history[1+prob.N_tot_levels:end,   1+n] .= vt
        uttvtt_history[1:prob.N_tot_levels,     1+n] .= utt
        uttvtt_history[1+prob.N_tot_levels:end, 1+n] .= vtt

        copy!(RHSu,u)
        axpy!(0.5*dt*weights[1],    ut,  RHSu)
        axpy!(0.25*dt^2*weights[2], utt, RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt*weights[1],    vt,  RHSv)
        axpy!(0.25*dt^2*weights[2], vtt, RHSv)

        copyto!(RHS, 1,                   RHSu, 1, prob.N_tot_levels)
        copyto!(RHS, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        t += dt

        IterativeSolvers.gmres!(uv, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+n+1] .= uv

        copyto!(u, 1, uv, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv, 1+prob.N_tot_levels, prob.N_tot_levels)
    end

    # One last time, for utvt history at final time
    utvt!(ut, vt, u, v, prob, controls, t, pcof)
    uttvtt!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

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
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}; return_time_derivatives=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_init_cond = size(prob.u0,2)
    uv_history = Array{Float64}(undef, prob.real_system_size, 1+prob.nsteps, N_init_cond)
    uv_and_derivatives_history = Array{Float64}(undef, prob.real_system_size, 1+prob.nsteps, 3, N_init_cond)

    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:N_init_cond

        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)

        # Call vector version of forward evolution
        if return_time_derivatives
            uv_and_derivatives_history[:,:,:,initial_condition_index] .= eval_forward_order4(
                vector_prob, controls, pcof, return_time_derivatives=true
            )
        else
            uv_history[:,:,initial_condition_index] .= eval_forward_order4(
                vector_prob, controls, pcof, return_time_derivatives=false
            )
        end
    end
    
    if return_time_derivatives
        return uv_and_derivatives_history
    else
        return uv_history
    end
end

function eval_forward_forced(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}, forcing_ary; order=2
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}
    if order == 2
        return eval_forward_forced_order2(prob, controls, pcof, forcing_ary)
    elseif order == 4
        return eval_forward_forced_order4(prob, controls, pcof, forcing_ary)
    end

    throw("Invalid Order: $order")
end

"""
Evolve schrodinger problem with forcing applied, and forcing given as an array
of forces at each discretized point in time.

Maybe I should also do a one with forcing functions as well.

Might be more convenient to merge this into the regular eval_forward functions,
with the forcing_ary defaulting to missing when we don't want to apply forcing.
"""
function eval_forward_forced_order2(
        prob::SchrodingerProb{M, V}, controls, pcof::V,
        forcing_ary::AbstractArray{Float64,3}, 
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}


    t = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(prob.real_system_size)
    copyto!(uv, 1,                   prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)
    uv_history = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64}   = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64}   = zeros(prob.N_tot_levels)
    RHS_uv::Vector{Float64} = zeros(prob.real_system_size)

    u = copy(prob.u0)
    v = copy(prob.v0)
    ut  = zeros(prob.N_tot_levels)
    vt  = zeros(prob.N_tot_levels)
    utt = zeros(prob.N_tot_levels)
    vtt = zeros(prob.N_tot_levels)


    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        # Careful, make certain that I can do this overwrite without messing up anything else
        copyto!(u, 1, uv_in, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv_in, 1+prob.N_tot_levels, prob.N_tot_levels)
        LHS_func!(
            uv_out, ut, vt, u, v,
            prob, controls, t, pcof, dt, prob.N_tot_levels
        )

        return nothing
    end

    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )


    for n in 0:prob.nsteps-1
        utvt!(ut, vt, u, v, prob, controls, t, pcof)

        copy!(RHSu, u)
        axpy!(0.5*dt, ut, RHSu)

        copy!(RHSv, v)
        axpy!(0.5*dt, vt, RHSv)

        copyto!(RHS_uv, 1,                   RHSu, 1, prob.N_tot_levels)
        copyto!(RHS_uv, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        axpy!(0.5*dt, forcing_ary[:, 1+n  , 1], RHS_uv)
        axpy!(0.5*dt, forcing_ary[:, 1+n+1, 1], RHS_uv)

        t += dt

        IterativeSolvers.gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)
        uv_history[:,1+n+1] .= uv

        copyto!(u, 1, uv, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv, 1+prob.N_tot_levels, prob.N_tot_levels)
    end
    
    return uv_history
end



"""
Forward evolution with forcing.

Right now the forcing array format is tightly coupled to the form of Schrodinger's
equation in the context of computing the gradient via the "forced" method. I
should make it less tightly coupled in the future, for clarity.
(so there wouldn't be a call to utvt! on the forcing array in this function)
"""
function eval_forward_forced_order4(
        prob::SchrodingerProb{M, V}, controls,  pcof::V, 
        forcing_ary::AbstractArray{Float64,3},
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    t = 0.0
    dt = prob.tf/prob.nsteps

    uv = zeros(prob.real_system_size)
    copyto!(uv, 1,                   prob.u0, 1, prob.N_tot_levels)
    copyto!(uv, 1+prob.N_tot_levels, prob.v0, 1, prob.N_tot_levels)

    uv_history = Matrix{Float64}(undef, prob.real_system_size, 1+prob.nsteps)
    uv_history[:,1] .= uv

    RHSu::Vector{Float64}   = zeros(prob.N_tot_levels)
    RHSv::Vector{Float64}   = zeros(prob.N_tot_levels)
    RHS_uv::Vector{Float64} = zeros(prob.real_system_size)

    u = copy(prob.u0)
    v = copy(prob.v0)
    ut  = zeros(prob.N_tot_levels)
    vt  = zeros(prob.N_tot_levels)
    utt = zeros(prob.N_tot_levels)
    vtt = zeros(prob.N_tot_levels)

    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        # Careful, make certain that I can do this overwrite without messing up anything else
        copyto!(u, 1, uv_in, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv_in, 1+prob.N_tot_levels, prob.N_tot_levels)
        return  LHS_func_order4!(
            uv_out, utt, vtt, ut, vt, u, v,
            prob, controls, t, pcof, dt, prob.N_tot_levels
        )
    end

    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )


    weights     = (1, 1/3)
    weights_LHS = (1, -1/3)
    for n in 0:prob.nsteps-1
        # First time derivative at current timestep
        utvt!(ut, vt, u, v, prob, controls, t, pcof)

        axpy!(1.0, forcing_ary[1:prob.N_tot_levels,     1+n, 1], ut)
        axpy!(1.0, forcing_ary[1+prob.N_tot_levels:end, 1+n, 1], vt)

        # Second time derivative at current timestep
        uttvtt!(utt, vtt, ut, vt, u, v, prob, controls, t, pcof)

        axpy!(1.0, forcing_ary[1:prob.N_tot_levels,     1+n, 2], utt)
        axpy!(1.0, forcing_ary[1+prob.N_tot_levels:end, 1+n, 2], vtt)

        # Accumulate RHS (current time contributions)
        copy!(RHSu, u)
        axpy!(0.5*dt*weights[1],    ut,  RHSu)
        axpy!(0.25*dt^2*weights[2], utt, RHSu)

        copy!(RHSv,v)
        axpy!(0.5*dt*weights[1],    vt,  RHSv)
        axpy!(0.25*dt^2*weights[2], vtt, RHSv)

        # Add in forcing from first time derivative at next timestep
        axpy!(0.5*dt*weights_LHS[1], forcing_ary[1:prob.N_tot_levels,     1+n+1, 1], RHSu)
        axpy!(0.5*dt*weights_LHS[1], forcing_ary[1+prob.N_tot_levels:end, 1+n+1, 1], RHSv)

        # Add in forcing from second time derivative at next timestep
        axpy!(0.25*dt^2*weights_LHS[2], forcing_ary[1:prob.N_tot_levels,     1+n+1,2], RHSu)
        axpy!(0.25*dt^2*weights_LHS[2], forcing_ary[1+prob.N_tot_levels:end, 1+n+1,2], RHSv)

        # Advance time to next point
        t += dt

        # Don't forget to differentiate the forcing from the first derivative
        utvt!(
            ut, vt, 
            forcing_ary[1:prob.N_tot_levels,     1+n+1, 1],
            forcing_ary[1+prob.N_tot_levels:end, 1+n+1, 1],
            prob, controls, t, pcof
        )

        axpy!(0.25*dt^2*weights_LHS[2], ut, RHSu)
        axpy!(0.25*dt^2*weights_LHS[2], vt, RHSv)

        copyto!(RHS_uv, 1,                   RHSu, 1, prob.N_tot_levels)
        copyto!(RHS_uv, 1+prob.N_tot_levels, RHSv, 1, prob.N_tot_levels)

        IterativeSolvers.gmres!(uv, LHS_map, RHS_uv, abstol=1e-15, reltol=1e-15)

        uv_history[:, 1+n+1] .= uv
        copyto!(u, 1, uv, 1,                   prob.N_tot_levels)
        copyto!(v, 1, uv, 1+prob.N_tot_levels, prob.N_tot_levels)
    end
    
    return uv_history
end
