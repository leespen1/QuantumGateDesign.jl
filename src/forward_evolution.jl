function eval_forward(
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{<: Real}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing,
        use_taylor_guess = true, verbose=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)

    eval_forward!(uv_history, prob, controls, pcof, order=order, forcing=forcing, use_taylor_guess=use_taylor_guess)

    return uv_history
end



function eval_forward!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{<: Real}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing,
        use_taylor_guess = true, verbose=false
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

        eval_forward!(
            this_uv_history, vector_prob, controls, pcof, order=order, forcing=this_forcing,
            use_taylor_guess=use_taylor_guess, verbose=verbose
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
function eval_forward!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{<: Real}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing,
        use_taylor_guess=true, verbose=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    t::Float64 = 0.0
    dt::Float64 = prob.tf/prob.nsteps
    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    # Allocate memory for storing u,v, and their derivatives at a single point in time
    uv_mat = zeros(prob.real_system_size, 1+N_derivatives)

    # Allocate memory for storing just u,v at a single point in time (to pass into/out of GMRES)
    uv_vec = zeros(prob.real_system_size)

    # Allocate memory for storing the right hand side (explicit part) of each timestep (to use as RHS of GMRES)
    RHS::Vector{Float64} = zeros(prob.real_system_size)

    # Allocate a matrix for storing the forcing at a single point in time (if we have forcing)
    if ismissing(forcing)
        forcing_mat = missing
        forcing_next_time_mat = missing
        forcing_helper_mat = missing
        forcing_helper_vec = missing 
    else
        forcing_mat = zeros(prob.real_system_size, N_derivatives)
        forcing_next_time_mat = zeros(prob.real_system_size, N_derivatives)
        forcing_helper_mat = zeros(prob.real_system_size, 1+N_derivatives) # For computing derivatives of forcing at next timestep
        forcing_helper_vec = zeros(prob.real_system_size) # For subtracting LHS forcing terms from the RHS (since they are explicit)
    end

    # This probably creates type-instability, as functions have singleton types, 
    # and this function is (I'm pretty sure) not created until runtime
    #
    # Create wrapper for computing action of LHS on uvₙ₊₁ at each timestep (to be compatible with LinearMaps)
    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        compute_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
        build_LHS!(uv_out, uv_mat, dt, N_derivatives)

        return nothing
    end

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        LHS_func_wrapper,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    N_gmres_iterations = 0

    gmres_iterable = IterativeSolvers.gmres_iterable!(zeros(prob.real_system_size), LHS_map, zeros(prob.real_system_size), abstol=1e-10, reltol=1e-10, restart=prob.real_system_size, initially_zero=false)

    # Important to do this after setting up the linear map and gmres_iterable. One of those seems to be overwriting uv_mat
    uv_mat[1:prob.N_tot_levels,                       1] .= prob.u0
    uv_mat[prob.N_tot_levels+1:prob.real_system_size, 1] .= prob.v0
    uv_history[:, :, 1] .= uv_mat


    # For some reason, using a Matrix as the preconditioner results in an error.
    #H_no_timing = [prob.system_asym prob.system_sym; -prob.system_sym prob.system_asym]
    #LHS_no_timing = Array(LinearAlgebra.diagm(ones(prob.real_system_size)) .- (dt .* H_no_timing))
    #LHS_no_timing = LinearAlgebra.diagm(ones(prob.real_system_size))

    # Perform the timesteps
    for n in 0:prob.nsteps-1

        # Compute the RHS (explicit part)
        t = n*dt
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n)
        end

        compute_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives, forcing_matrix=forcing_mat)
        uv_history[:, :, 1+n] .= uv_mat
        build_RHS!(RHS, uv_mat, dt, N_derivatives)

        if use_taylor_guess
            taylor_expand!(uv_vec, uv_mat, dt, N_derivatives) # Use taylor expansion as guess
        else
            uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        end

        # Use GMRES to perform the timestep (implicit part)
        t = (n+1)*dt

        # Account for forcing from next timestep
        if !ismissing(forcing_next_time_mat)
            forcing_next_time_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n+1)
            forcing_helper_mat .= 0
            compute_derivatives!(forcing_helper_mat, prob, controls, t, pcof, N_derivatives, forcing_matrix=forcing_next_time_mat)
            build_LHS!(forcing_helper_vec, forcing_helper_mat, dt, N_derivatives)
            axpy!(-1.0, forcing_helper_vec, RHS)
        end


        update_gmres_iterable!(gmres_iterable, uv_vec, RHS)

        for iter in gmres_iterable
            N_gmres_iterations += 1
        end
        #IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-10, reltol=1e-10, restart=prob.real_system_size, initially_zero=false)
        #~, history = IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-15, reltol=1e-15, restart=prob.real_system_size, log=true, initially_zero=false)
        #N_gmres_iterations += history.iters


        uv_mat[:,1] .= gmres_iterable.x
    end
    N_gmres_iterations /= prob.nsteps
    if verbose
        println("Average # of gmres iterations: ", N_gmres_iterations)
    end

    # Compute the derivatives of uv at the final time and store them
    t = prob.nsteps*dt
    if !ismissing(forcing_mat)
        forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+prob.nsteps)
    end
    compute_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
    uv_history[:, :, 1+prob.nsteps] .= uv_mat

    return nothing
end


"""
Evolve a vector SchrodingerProblem forward in time. Return the history of the
state vector (u/v) in a 3-index array, where the first index of
corresponds to the vector component, the second index corresponds to the
derivative to be taken, and the third index corresponds to the timestep number.
"""
function eval_forward(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{<: Real}; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing,
        use_taylor_guess=true, verbose=false
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    N_derivatives = div(order, 2)

    # Allocate memory for storing u,v, and their derivatives over all points in time
    uv_history = Array{Float64, 3}(undef, prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    eval_forward!(
        uv_history, prob, controls, pcof, order=order, forcing=forcing,
        use_taylor_guess=use_taylor_guess, verbose=verbose
    )

    return uv_history
end


"""
Evolve a vector SchrodingerProblem forward in time. Return the history of the
state vector (u/v) in a 3-index array, where the first index of
corresponds to the vector component, the second index corresponds to the
derivative to be taken, and the third index corresponds to the timestep number.
"""
function eval_adjoint(
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{<: Real}, terminal_condition::AbstractVector{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 2}, Missing}=missing
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}}

    N_derivatives = div(order, 2)

    # Allocate memory for storing u,v, and their derivatives over all points in time
    #uv_history = Array{Float64, 3}(undef, prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    eval_adjoint!(uv_history, prob, controls, pcof, terminal_condition, order=order, forcing=forcing)

    return uv_history
end

function eval_adjoint!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V}, controls,
        pcof::AbstractVector{<: Real}, terminal_condition::AbstractVector{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 2}, Missing}=missing
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

    # This probably creates type-instability, as functions have singleton types, 
    # and this function is (I'm pretty sure) not created until runtime
    #
    # Create wrapper for computing action of LHS on uvₙ₊₁ at each timestep (to be compatible with LinearMaps)
    function LHS_func_wrapper(uv_out::AbstractVector{Float64}, uv_in::AbstractVector{Float64})
        uv_mat[:,1] .= uv_in
        compute_adjoint_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
        # Do I need to make and adjoint version of this? I don't think so, considering before LHS only used adjoint for utvt!, not the quadrature
        # But maybe there is a negative t I need to worry about. Maybe just provide dt as -dt
        build_LHS!(uv_out, uv_mat, dt, N_derivatives)

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

        compute_adjoint_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
        uv_history[:, :, 1+n] .= uv_mat
        build_RHS!(RHS, uv_mat, dt, N_derivatives)

        # Add forcing. It doesn't go into the derivative calculation like it
        # does in the forward evolution. Index should be that of the vector we are trying to assign to in the history.
        if !ismissing(forcing)
            RHS .+= view(forcing, :, 1+n-1) 
        end

        # Use GMRES to perform the timestep (implicit part)
        uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-15, reltol=1e-15, restart=prob.real_system_size)
        uv_mat[:,1] .= uv_vec
    end

    # Compute the derivatives of uv at n=1 and store them
    t = dt
    compute_adjoint_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
    uv_history[:, :, 2] .= uv_mat

    return nothing
end



function eval_adjoint(
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{<: Real}, terminal_condition::AbstractMatrix{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_ess_levels)

    eval_adjoint!(uv_history, prob, controls, pcof, terminal_condition,
        order=order, forcing=forcing
    )

    return uv_history
end


function eval_adjoint!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2}, controls,
        pcof::AbstractVector{<: Real}, terminal_condition::AbstractMatrix{Float64}
        ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing
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
            this_forcing = @view forcing[:, :, initial_condition_index]
        end

        eval_adjoint!(
            this_uv_history, vector_prob, controls, pcof, terminal_condition_vec,
            order=order, forcing=this_forcing
        )
    end
end

"""
Work in progress, trying to change from inline function
"""
mutable struct TimestepHolder{T1, T2, T3, T4}
    N_derivatives::Int64
    uv_mat::Matrix{Float64}
    uv_vec::Vector{Float64}
    pcof::Vector{Float64}
    controls::T1
    prob::T2
    lhs_holder::T3
    LHS_map::T4
    RHS::Matrix{Float64}
    function TimestepHolder(system_size, N_derivatives, pcof, controls, prob)
        uv_mat = zeros(system_size, 1+N_derivatives) 
        uv_vec = zeros(system_size) 
        RHS = zeros(system_size) 
        lhs_holder = LHSHolder(t, dt, N_derivatives, uv_mat, pcof, controls, prob)
        # Create linear map out of LHS_func_wrapper, to use in GMRES
        LHS_map = LinearMaps.LinearMap(
            lhs_holder,
            prob.real_system_size, prob.real_system_size,
            ismutating=true
        )
        new{T1, T2, T3, T4}(N_derivatives, uv_mat, uv_vec, pcof, controls, prob, lhs_holder, LHS_map, RHS)
    end
end


mutable struct LHSHolder{T1, T2}
    tnext::Float64
    dt::Float64
    N_derivatives::Int64
    uv_mat::Matrix{Float64}
    pcof::Vector{Float64}
    controls::T1
    prob::T2
end

"""
WOrk in progress, callable struct
"""
function (self::LHSHolder)(out_vec, in_vec)
    self.uv_mat[:,1] .= in_vec
    compute_derivatives!(self.uv_mat, self.prob, self.controls, self.tnext, self.pcof, self.N_derivatives)
    build_LHS!(out_vec, self.uv_mat, self.dt, self.N_derivatives)

    return nothing
end

function perform_timestep(timestep_holder::TimestepHolder, t, dt)
    TimestepHolder.tnext = t+dt
    TimestepHolder.dt = dt

    #if !ismissing(forcing_mat)
    #    forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n)
    #end
    forcing_mat = missing

    compute_derivatives!(timestep_holder.uv_mat, timestep_holder.prob, timestep_holder.controls,
                         t, timestep_holder.pcof, timestep_holder.N_derivatives, forcing_matrix=forcing_mat)
    #uv_history[:, :, 1+n] .= uv_mat
    
    build_RHS!(timestep_holder.RHS, timestep_holder.uv_mat, dt, timestep_holder.N_derivatives)

    if use_taylor_guess
        taylor_expand!(timestep_holder.uv_vec, timestep_holder.uv_mat, dt, timestep_holder.N_derivatives) # Use taylor expansion as guess
    else
        uv_vec .= view(timestep_holder.uv_mat, 1:timestep_holder.prob.real_system_size, 1) # Use current timestep as initial guess for gmres
    end

    ## Account for forcing from next timestep
    #if !ismissing(forcing_next_time_mat)
    #    forcing_next_time_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n+1)
    #    forcing_helper_mat .= 0
    #    compute_derivatives!(forcing_helper_mat, prob, controls, t, pcof, N_derivatives, forcing_matrix=forcing_next_time_mat)
    #    build_LHS!(forcing_helper_vec, forcing_helper_mat, dt, N_derivatives)
    #    axpy!(-1.0, forcing_helper_vec, RHS)
    #end

    IterativeSolvers.gmres!(uv_vec, timestep_holder.LHS_map, timestep_holder.RHS, abstol=1e-15, reltol=1e-15, restart=prob.real_system_size, initially_zero=false)
    #~, history = IterativeSolvers.gmres!(uv_vec, LHS_map, RHS, abstol=1e-15, reltol=1e-15, restart=prob.real_system_size, log=true, initially_zero=false)
    #N_gmres_iterations += history.iters

    timestep_holder.uv_mat[:,1] .= timestep_holder.uv_vec

    return nothing
end



function jacobi!(x, LHS_map, RHS, tol=1e-10)
    xn = copy(x)
    xnp1 = copy(x)

    err = tol + 1
    n_iterations = 0
    # x^(n+1) = (b + x^n - B*x^n)
    while err > tol
         xn .= xnp1
         LinearAlgebra.axpy!(1, RHS, xnp1)
         mul!(xnp1, LHS_map, xn, -1, 1)
         err = LinearAlgebra.norm(xn .- xnp1)/norm(xnp1)
         n_iterations += 1
    end

    x .= xnp1

    return n_iterations
end 

function jacobi_real!(x, LHS_map, RHS, tol=1e-10)
    xn = copy(x)
    xnp1 = copy(x)

    err = tol + 1
    n_iterations = 0
    # x^(n+1) = (b + x^n - B*x^n)
    while err > tol
         xn .= xnp1
         LinearAlgebra.axpy!(1, RHS, xnp1)
         mul!(xnp1, LHS_map, xn, -1, 1)
         err = LinearAlgebra.norm(xn .- xnp1)/norm(xnp1)
         n_iterations += 1
    end

    x .= xnp1

    return n_iterations
end 

function update_gmres_iterable!(iterable, x, b)
    iterable.b .= b
    iterable.x .= x
    iterable.mv_products = 0
    iterable.arnoldi.H .= 0
    iterable.arnoldi.V .= 0
    iterable.residual.accumulator = 1
    iterable.residual.current = 1
    iterable.residual.nullvec .= 1
    iterable.residual.β = 1
    iterable.residual.current = IterativeSolvers.init!(
        iterable.arnoldi, iterable.x, iterable.b, iterable.Pl, iterable.Ax,
        initially_zero=false
    )
    iterable.residual.nullvec .= 1
    IterativeSolvers.init_residual!(iterable.residual, iterable.residual.current)
    iterable.β = iterable.residual.current
    return nothing
end
