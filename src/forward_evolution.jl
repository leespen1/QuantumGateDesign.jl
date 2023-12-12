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

function eval_forward_arbitrary_order(
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)

    eval_forward_arbitrary_order!(uv_history, prob, controls, pcof, order=order, forcing=forcing)

    return uv_history
end


function eval_forward_arbitrary_order!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)


    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:prob.N_ess_levels
        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)
        this_uv_history = @view uv_history[:, :, :, initial_condition_index]

        if ismissing(forcing)
            this_forcing = missing
        else
            this_forcing = @view forcing[:, :, :, initial_condition_index]
        end

        eval_forward_arbitrary_order!(
            this_uv_history, vector_prob, controls, pcof, order=order, forcing=this_forcing
        )
    end
end

"""
Evolve a vector SchrodingerProblem forward in time. Store the history of the
state vector (u/v) in the array uv_history. The first index of uv_history
corresponds to the vector component, the second index corresponds to the
derivative to be taken, and the third index corresponds to the timestep number.

E.g. uv_history[:,1,1] is the initial condition, uv_history[:,2,1] is the value
of du/dt and dv/dt at t=0, uv_history[:,1,end] is the value of uv at t=tf, etc.

Currently tested against old implementations, and for a small example gave the
same results to near machine precision. The maximum difference between entries
in the histories created by the old and new implementations was 1e-13. It seemed
like most entries differed by less than 1e-14.

I plan to make an adjoint version of this.
"""
function eval_forward_arbitrary_order!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    
    t = 0.0
    dt = prob.tf/prob.nsteps
    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    # Allocate memory for storing u,v, and their derivatives at a single point in time
    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
    uv_mat[1:prob.N_tot_levels,                       1] .= prob.u0
    uv_mat[prob.N_tot_levels+1:prob.real_system_size, 1] .= prob.v0

    uv_history[:, :, 1] .= uv_mat

    # Allocate memory for storing just u,v at a single point in time (to pass into/out of GMRES)
    uv_vec = zeros(prob.real_system_size)
    # Allocate memory for storing the right hand side (explicit part) of each timestep (to use as RHS of GMRES)
    RHS::Vector{Float64} = zeros(prob.real_system_size)

    # Allocate a matrix for storing the forcing at a single point in time (if we have forcing)
    if ismissing(forcing)
        forcing_mat = missing
    else
        forcing_mat = zeros(prob.real_system_size, N_derivatives)
    end

    # This probably creates type-instability, as functions have singleton types, 
    # and this function is (I'm pretty sure) not created until runtime
    #
    # Create wrapper for computing action of LHS on uvₙ₊₁ at each timestep (to be compatible with LinearMaps)
    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives)
        arbitrary_LHS!(uv_out, uv_mat, dt, N_derivatives)

        return nothing
    end

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    # Perform the timesteps
    for n in 0:prob.nsteps-1

        # Compute the RHS (explicit part)
        t = n*dt
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n)
        end

        arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives)
        uv_history[:, :, 1+n] .= uv_mat
        arbitrary_RHS!(RHS, uv_mat, dt, N_derivatives)


        # Use GMRES to perform the timestep (implicit part)
        t = (n+1)*dt
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n+1)
        end

        uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
        uv_mat[:,1] .= uv_vec
    end

    # Compute the derivatives of uv at the final time and store them
    t = prob.nsteps*dt
    if !ismissing(forcing_mat)
        forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+prob.nsteps)
    end
    arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives)
    uv_history[:, :, 1+prob.nsteps] .= uv_mat

    return nothing
end


"""
Evolve a vector SchrodingerProblem forward in time. Return the history of the
state vector (u/v) in a 3-index array, where the first index of
corresponds to the vector component, the second index corresponds to the
derivative to be taken, and the third index corresponds to the timestep number.
"""
function eval_forward_arbitrary_order(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    N_derivatives = div(order, 2)

    # Allocate memory for storing u,v, and their derivatives over all points in time
    uv_history = Array{Float64, 3}(undef, prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    eval_forward_arbitrary_order!(uv_history, prob, controls, pcof, order=order, forcing=forcing)

    return uv_history
end


"""
Evolve a vector SchrodingerProblem forward in time. Return the history of the
state vector (u/v) in a 3-index array, where the first index of
corresponds to the vector component, the second index corresponds to the
derivative to be taken, and the third index corresponds to the timestep number.
"""
function eval_adjoint_arbitrary_order(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}, terminal_condition::AbstractVector{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    N_derivatives = div(order, 2)

    # Allocate memory for storing u,v, and their derivatives over all points in time
    #uv_history = Array{Float64, 3}(undef, prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    eval_adjoint_arbitrary_order!(uv_history, prob, controls, pcof, terminal_condition, order=order, forcing=forcing)

    return uv_history
end

function eval_adjoint_arbitrary_order!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{Float64}, terminal_condition::AbstractVector{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    
    t = 0.0
    dt = prob.tf/prob.nsteps
    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    # Allocate memory for storing u,v, and their derivatives at a single point in time
    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
    uv_mat[1:prob.N_tot_levels,                       1] .= terminal_condition[1:prob.N_tot_levels]
    uv_mat[prob.N_tot_levels+1:prob.real_system_size, 1] .= terminal_condition[prob.N_tot_levels+1:prob.real_system_size]

    uv_history[:, :, 1+prob.nsteps] .= uv_mat

    # Allocate memory for storing just u,v at a single point in time (to pass into/out of GMRES)
    uv_vec = zeros(prob.real_system_size)
    # Allocate memory for storing the right hand side (explicit part) of each timestep (to use as RHS of GMRES)
    RHS::Vector{Float64} = zeros(prob.real_system_size)

    # Allocate a matrix for storing the forcing at a single point in time (if we have forcing)
    if ismissing(forcing)
        forcing_mat = missing
    else
        forcing_mat = zeros(prob.real_system_size, N_derivatives)
    end

    # This probably creates type-instability, as functions have singleton types, 
    # and this function is (I'm pretty sure) not created until runtime
    #
    # Create wrapper for computing action of LHS on uvₙ₊₁ at each timestep (to be compatible with LinearMaps)
    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives, use_adjoint=true)
        # Do I need to make and adjoint version of this? I don't think so, considering before LHS only used adjoint for utvt!, not the quadrature
        # But maybe there is a negative t I need to worry about. Maybe just provide dt as -dt
        arbitrary_LHS!(uv_out, uv_mat, dt, N_derivatives)

        return nothing
    end

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    # Perform the timesteps
    for n in prob.nsteps:-1:2

        # Compute the RHS (explicit part). 
        # NOTE THAT SAME TIME IS USED FOR EXPLICIT AND IMPLICIT PART, UNLIKE IN FORWARD EVOLUTION
        t = (n-1)*dt
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n)
        end

        arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives, use_adjoint=true)
        uv_history[:, :, 1+n] .= uv_mat
        arbitrary_RHS!(RHS, uv_mat, dt, N_derivatives)


        # Use GMRES to perform the timestep (implicit part)
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n-1)
        end

        uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-15, reltol=1e-15)
        uv_mat[:,1] .= uv_vec
    end

    # Compute the derivatives of uv at the final time and store them
    t = dt
    if !ismissing(forcing_mat)
        forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+prob.nsteps)
    end
    arbitrary_order_uv_derivative!(uv_mat, prob, controls, t, pcof, N_derivatives, use_adjoint=true)
    uv_history[:, :, 2] .= uv_mat

    return nothing
end

function eval_adjoint_arbitrary_order(
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}, terminal_condition::AbstractMatrix{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)

    eval_adjoint_arbitrary_order!(uv_history, prob, controls, pcof, terminal_condition, order=order, forcing=forcing)

    return uv_history
end


function eval_adjoint_arbitrary_order!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{Float64}, terminal_condition::AbstractMatrix{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)


    # Handle i-th initial condition (THREADS HERE)
    for initial_condition_index=1:prob.N_ess_levels
        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)
        terminal_condition_vec = @view terminal_condition[:, initial_condition_index]
        this_uv_history = @view uv_history[:, :, :, initial_condition_index]

        if ismissing(forcing)
            this_forcing = missing
        else
            this_forcing = @view forcing[:, :, :, initial_condition_index]
        end

        eval_adjoint_arbitrary_order!(
            this_uv_history, vector_prob, controls, pcof, order=order, forcing=this_forcing
        )
    end
end

