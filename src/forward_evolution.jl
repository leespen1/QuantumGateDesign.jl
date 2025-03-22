"""
Type for keeping track of how many GMRES iterations occured throughout the 
forward solves.
"""
mutable struct GMRESTracker
    N_linear_solves::Int64
    N_converged::Int64
    total_N_iterations::Int64
    max_N_iterations::Int64
    accumulated_residuals::Float64
    max_residual::Float64
    function GMRESTracker()
        new(0, 0, 0, -1, 0.0, 0.0)
    end
end

# TODO use functions for averages  
@inline function avg_residual(tracker::GMRESTracker)
    return tracker.accumulated_residuals / tracker.N_linear_solves
end

@inline function avg_N_iterations(tracker::GMRESTracker)
    return tracker.total_N_iterations / tracker.N_linear_solves

end


function update_gmres_tracker!(tracker::GMRESTracker, N_iter::Integer, residual::Real, converged::Bool)
    tracker.N_linear_solves += 1
    tracker.N_converged += converged ? 1 : 0
    tracker.total_N_iterations += N_iter
    tracker.max_N_iterations = max(N_iter, tracker.max_N_iterations)
    tracker.accumulated_residuals += residual
    tracker.max_residual = max(residual, tracker.max_residual)
    return tracker
end


function merged_gmres_tracker(trackers::Vararg{GMRESTracker})
    merged_tracker = GMRESTracker()
    for tracker in trackers
        merge_gmres_trackers!(merged_tracker, tracker)
    end
    return merged_tracker
end

function merge_gmres_trackers!(gt1::GMRESTracker, gt2::GMRESTracker)
    gt1.N_linear_solves += gt2.N_linear_solves
    gt1.N_converged += gt2.N_converged
    gt1.total_N_iterations += gt2.total_N_iterations
    gt1.max_N_iterations = max(gt1.max_N_iterations, gt2.max_N_iterations)
    gt1.accumulated_residuals += gt2.accumulated_residuals
    gt1.max_residual = max(gt1.max_residual, gt2.max_residual)

    return gt1
end

function copyto_gmres_tracker!(gt1::GMRESTracker, gt2::GMRESTracker)
    gt1.N_linear_solves = gt2.N_linear_solves
    gt1.N_converged = gt2.N_converged
    gt1.total_N_iterations = gt2.total_N_iterations
    gt1.max_N_iterations = gt2.max_N_iterations
    gt1.accumulated_residuals = gt2.accumulated_residuals
    gt1.max_residual = gt2.max_residual

    return gt1
end

"""
    eval_forward(prob, controls, pcof; [order=2, saveEveryNsteps=1, forcing=missing,])

Simulate a `SchrodingerProb` forward in time. Return the history of the state
vector for each initial condition as a 4D array.

# Arguments
- `prob::SchrodingerProb`: Object containing the Hamiltonians, number of timesteps, etc.
- `controls`: An `AstractControl` or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.
- `pcof::AbstractVector{<: Real}`: The control vector.
- `order::Int64=2`: Which order of the method to use.
- `saveEveryNsteps::Int64=1`: Only store the state every `saveEveryNsteps` timesteps.
- `forcing::Union{AbstractArray{Float64}, Missing}`: Optional forcing array, ordered in same format as the returned history.
"""
function eval_forward(
        prob::SchrodingerProb{M1, M2, P}, controls::ControlsType, pcof::AbstractVector{<: Real};
        order::Int=2, saveEveryNsteps::Int=1,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing, verbose::Bool=false,
        gmres_tracker::Union{GMRESTracker, Missing}=missing,
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, P}

    N_derivatives = div(order, 2)
    nsteps_save = div(prob.nsteps, saveEveryNsteps)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+nsteps_save, prob.N_initial_conditions)

    eval_forward!(uv_history, prob, controls, pcof, order=order,
                  saveEveryNsteps=saveEveryNsteps; forcing=forcing,
                  verbose=verbose, gmres_tracker=gmres_tracker)

    return real_to_complex(uv_history[:,1,:,:])
end



function eval_forward!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2, P}, controls::ControlsType,
        pcof::AbstractVector{<: Real}; order::Int=2, saveEveryNsteps::Int=1,
        forcing::Union{AbstractArray{Float64, 4}, Missing}=missing,
        verbose::Bool=false, gmres_tracker::Union{GMRESTracker, Missing}=missing
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, P}

    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    nsteps_save = div(prob.nsteps, saveEveryNsteps)
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+nsteps_save, prob.N_initial_conditions)


    # Handle i-th initial condition (THREADS HERE)
    gmres_trackers = Vector{GMRESTracker}(undef, prob.N_initial_conditions)
    Threads.@threads for initial_condition_index=1:prob.N_initial_conditions
        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)
        controls_copy = deepcopy(controls) # Make copies of control, so that they work with multithreading

        this_uv_history = @view uv_history[:, :, :, initial_condition_index]

        if ismissing(forcing)
            this_forcing = missing
        else
            this_forcing = @view forcing[:, :, :, initial_condition_index]
        end

        local_gmres_tracker = eval_forward!(
            this_uv_history, vector_prob, controls_copy, pcof; order=order, 
            saveEveryNsteps=saveEveryNsteps, forcing=this_forcing
        )
        gmres_trackers[initial_condition_index] = local_gmres_tracker
    end

    full_gmres_tracker = merged_gmres_tracker(gmres_trackers...)
    if verbose && (full_gmres_tracker.N_converged != full_gmres_tracker.N_linear_solves)
        @warn "Only $(full_gmres_tracker.N_converged)/$(full_gmres_tracker.N_linear_solves) GMRES linear solves converged."
    end

    if !ismissing(gmres_tracker)
        copyto_gmres_tracker!(gmres_tracker, full_gmres_tracker)
    end

    return full_gmres_tracker
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
        prob::SchrodingerProb{M, V, P}, controls::ControlsType,
        pcof::AbstractVector{<: Real}; order::Int=2, saveEveryNsteps::Int=1,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing,
        use_taylor_guess=true,
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, P}


    gmres_tracker = GMRESTracker()

    t = 0.0
    dt = prob.tf/prob.nsteps
    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    nsteps_save = div(prob.nsteps, saveEveryNsteps)
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+nsteps_save)

    # Allocate memory for storing u,v, and their derivatives at a single point in time
    uv_mat = Matrix{Float64}(undef, prob.real_system_size, 1+N_derivatives)
    # Allocate memory for storing just u,v at a single point in time (to pass into/out of GMRES)
    uv_vec = Vector{Float64}(undef, prob.real_system_size)
    # Allocate memory for storing the right hand side (explicit part) of each timestep (to use as RHS of GMRES)
    RHS = Vector{Float64}(undef, prob.real_system_size)

    uv_mat .= 0
    uv_vec .= 0
    RHS .= 0

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

    lhs_holder = LHSHolder(prob, N_derivatives, dt)

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        lhs_holder,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    Pl = P(prob, order, false)

    gmres_iterable = IterativeSolvers.gmres_iterable!(
        zeros(prob.real_system_size), LHS_map, zeros(prob.real_system_size),
        abstol=prob.gmres_abstol, reltol=prob.gmres_reltol, restart=prob.real_system_size,
        initially_zero=false, Pl=Pl, maxiter=prob.real_system_size
    )

    # Important to do this after setting up the linear map and gmres_iterable. One of those seems to be overwriting uv_mat
    uv_mat[1:prob.N_tot_levels,                       1] .= prob.u0
    uv_mat[prob.N_tot_levels+1:prob.real_system_size, 1] .= prob.v0
    uv_history[:, :, 1] .= uv_mat

    # Get control function values for the initial time
    t = 0.0
    fill_p_mat!(lhs_holder.control_vals_real, controls, t, pcof) 
    fill_q_mat!(lhs_holder.control_vals_imag, controls, t, pcof) 

    # Perform the timesteps
    for n in 0:prob.nsteps-1

        # Compute the RHS (explicit part)
        t = n*dt
        if !ismissing(forcing_mat)
            forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n)
        end

        # Reuse the control function values from the implicit part of the previous timestep
        compute_derivatives!(
            uv_mat, prob, lhs_holder.control_vals_real, lhs_holder.control_vals_imag,
            N_derivatives, forcing_matrix=forcing_mat
        )

        if ((n % saveEveryNsteps) == 0)
            uv_history[:, :, 1+div(n, saveEveryNsteps)] .= uv_mat
        end

        build_RHS!(RHS, uv_mat, dt, N_derivatives)

        if use_taylor_guess
            taylor_expand!(uv_vec, uv_mat, dt, N_derivatives) # Use taylor expansion as guess
        else
            uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        end

        # Use GMRES to perform the timestep (implicit part)
        t = (n+1)*dt
        # Get control function values for the next time (the implicit evaluation)
        fill_p_mat!(lhs_holder.control_vals_real, controls, t, pcof) 
        fill_q_mat!(lhs_holder.control_vals_imag, controls, t, pcof) 

        # Account for forcing from next timestep (which is still explicit)
        if !ismissing(forcing_next_time_mat)
            forcing_next_time_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+n+1)
            forcing_helper_mat .= 0
            compute_derivatives!(
                forcing_helper_mat, prob, lhs_holder.control_vals_real,
                lhs_holder.control_vals_imag, N_derivatives,
                forcing_matrix=forcing_next_time_mat
            )
            build_LHS!(forcing_helper_vec, forcing_helper_mat, dt, N_derivatives)
            axpy!(-1.0, forcing_helper_vec, RHS)
        end

        update_gmres_iterable!(gmres_iterable, uv_vec, RHS)

        N_gmres_iterations = 0
        for iter in gmres_iterable
            N_gmres_iterations += 1
        end

        update_gmres_tracker!(gmres_tracker, N_gmres_iterations,
                              gmres_iterable.residual.current,
                              IterativeSolvers.converged(gmres_iterable))

        uv_mat[:,1] .= gmres_iterable.x
    end

    # Compute the derivatives of uv at the final time and store them
    t = prob.nsteps*dt
    if !ismissing(forcing_mat)
        forcing_mat .= view(forcing, 1:prob.real_system_size, 1:N_derivatives, 1+prob.nsteps)
    end
    compute_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives)
    ## Should it actually be the following instead?
    #compute_derivatives!(uv_mat, prob, controls, t, pcof, N_derivatives, forcing_matrix=forcing_mat)

    if ((prob.nsteps % saveEveryNsteps) == 0)
        uv_history[:, :, 1+div(prob.nsteps, saveEveryNsteps)] .= uv_mat
    end

    return gmres_tracker
end



function eval_adjoint(
        prob::SchrodingerProb{M1, M2, P}, controls::ControlsType,
        pcof::AbstractVector{<: Real}, terminal_condition::AbstractMatrix{Float64};
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing,
        order::Int=2, verbose::Bool=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}, P}

    N_derivatives = div(order, 2)
    uv_history = zeros(prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)

    eval_adjoint!(uv_history, prob, controls, pcof, terminal_condition;
        order=order, forcing=forcing, verbose=verbose
    )

    return uv_history
end

function eval_adjoint!(uv_history::AbstractArray{Float64, 4},
        prob::SchrodingerProb{M1, M2}, controls::ControlsType,
        pcof::AbstractVector{<: Real},
        terminal_condition::AbstractMatrix{Float64} ; order::Int=2,
        forcing::Union{AbstractArray{Float64, 3}, Missing}=missing,
        verbose::Bool=false
    ) where {M1<:AbstractMatrix{Float64}, M2<:AbstractMatrix{Float64}}

    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps, prob.N_initial_conditions)


    # Handle i-th initial condition (THREADS HERE)
    gmres_trackers = Vector{GMRESTracker}(undef, prob.N_initial_conditions)
    Threads.@threads for initial_condition_index=1:prob.N_initial_conditions
        vector_prob = VectorSchrodingerProb(prob, initial_condition_index)
        controls_copy = deepcopy(controls)

        terminal_condition_vec = @view terminal_condition[:, initial_condition_index]
        this_uv_history = @view uv_history[:, :, :, initial_condition_index]

        if ismissing(forcing)
            this_forcing = missing
        else
            this_forcing = @view forcing[:, :, initial_condition_index]
        end

        gmres_tracker = eval_adjoint!(
            this_uv_history, vector_prob, controls_copy, pcof, terminal_condition_vec;
            order=order, forcing=this_forcing
        )
        gmres_trackers[initial_condition_index] = gmres_tracker
    end

    full_gmres_tracker = reduce(merge, gmres_trackers)
    if verbose && (full_gmres_tracker.N_converged != full_gmres_tracker.N_linear_solves)
        @warn "Only $(full_gmres_tracker.N_converged)/$(full_gmres_tracker.N_linear_solves) GMRES linear solves converged."
    end

    return full_gmres_tracker
end



function eval_adjoint!(uv_history::AbstractArray{Float64, 3},
        prob::SchrodingerProb{M, V, P}, controls::ControlsType,
        pcof::AbstractVector{<: Real},
        terminal_condition::AbstractVector{Float64};
        forcing::Union{AbstractArray{Float64, 2}, Missing}=missing,
        order::Int=2, use_taylor_guess=true, verbose::Bool=false,
    ) where {M<:AbstractMatrix{Float64}, V<:AbstractVector{Float64}, P}

    gmres_tracker = GMRESTracker()
    
    t = 0.0
    dt = prob.tf/prob.nsteps
    N_derivatives = div(order, 2)

    # Check size of uv_history storage
    @assert size(uv_history) == (prob.real_system_size, 1+N_derivatives, 1+prob.nsteps)

    # Allocate memory for storing u,v, and their derivatives at a single point in time
    uv_mat = Matrix{Float64}(undef, prob.real_system_size, 1+N_derivatives)
    # Allocate memory for storing just u,v at a single point in time (to pass into/out of GMRES)
    uv_vec = Vector{Float64}(undef, prob.real_system_size)
    # Allocate memory for storing the right hand side (explicit part) of each timestep (to use as RHS of GMRES)
    RHS = Vector{Float64}(undef, prob.real_system_size)

    uv_mat .= 0
    uv_vec .= 0
    RHS .= 0

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


    lhs_holder = LHSHolderAdjoint(prob, N_derivatives, dt)

    # Create linear map out of LHS_func_wrapper, to use in GMRES
    LHS_map = LinearMaps.LinearMap(
        lhs_holder,
        prob.real_system_size, prob.real_system_size,
        ismutating=true
    )

    Pl = P(prob, order, true)

    gmres_iterable = IterativeSolvers.gmres_iterable!(
        zeros(prob.real_system_size), LHS_map, zeros(prob.real_system_size),
        abstol=prob.gmres_abstol, reltol=prob.gmres_reltol, restart=prob.real_system_size,
        initially_zero=false, Pl=Pl
    )

    # Important to do this after setting up the linear map and gmres_iterable. One of those seems to be overwriting uv_mat
    uv_mat[1:prob.N_tot_levels,                       1] .= view(terminal_condition,1:prob.N_tot_levels)
    uv_mat[prob.N_tot_levels+1:prob.real_system_size, 1] .= view(terminal_condition,prob.N_tot_levels+1:prob.real_system_size)

    uv_history[:, :, 1+prob.nsteps] .= uv_mat

    # Perform the timesteps
    for n in prob.nsteps:-1:2
        # Compute the RHS (explicit part)
        t = (n-1)*dt
        fill_p_mat!(lhs_holder.control_vals_real, controls, t, pcof) 
        fill_q_mat!(lhs_holder.control_vals_imag, controls, t, pcof) 

        compute_adjoint_derivatives!(
            uv_mat, prob, lhs_holder.control_vals_real,
            lhs_holder.control_vals_imag, N_derivatives,
            lhs_holder.working_vec,
            lhs_holder.working_matrix, lhs_holder.working_matrix2
        )
        uv_history[:, :, 1+n] .= uv_mat
        build_RHS!(RHS, uv_mat, dt, N_derivatives)

        # Add forcing. It doesn't go into the derivative calculation like it
        # does in the forward evolution. Index should be that of the vector we are trying to assign to in the history.
        if !ismissing(forcing)
            RHS .+= view(forcing, :, 1+n-1) 
        end

        # Not sure how valid the taylor guess is here, since the ODE is not the same
        if use_taylor_guess
            taylor_expand!(uv_vec, uv_mat, -dt, N_derivatives) # Use (backward) taylor expansion as guess
        else
            uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        end

        # Use GMRES to perform the timestep (implicit part)
        uv_vec .= view(uv_mat, 1:prob.real_system_size, 1) # Use current timestep as initial guess for gmres
        update_gmres_iterable!(gmres_iterable, uv_vec, RHS)

        N_gmres_iterations = 0
        for iter in gmres_iterable
            N_gmres_iterations += 1
        end

        update_gmres_tracker!(gmres_tracker, N_gmres_iterations,
                              gmres_iterable.residual.current,
                              IterativeSolvers.converged(gmres_iterable))

        uv_mat[:,1] .= gmres_iterable.x
    end

    # Compute the derivatives of uv at n=1 and store them
    t = dt
    fill_p_mat!(lhs_holder.control_vals_real, controls, t, pcof) 
    fill_q_mat!(lhs_holder.control_vals_imag, controls, t, pcof) 
    compute_adjoint_derivatives!(
        uv_mat, prob, lhs_holder.control_vals_real, lhs_holder.control_vals_imag,
        N_derivatives, lhs_holder.working_vec, lhs_holder.working_matrix,
        lhs_holder.working_matrix2
    )
    uv_history[:, :, 2] .= uv_mat

    return gmres_tracker
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



struct LHSHolder{T}
    N_derivatives::Int64
    dt::Float64
    uv_mat::Matrix{Float64}
    control_vals_real::Matrix{Float64}
    control_vals_imag::Matrix{Float64}
    prob::T
    function LHSHolder(prob, N_derivatives, dt)
        uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
        control_vals_real = zeros(1+N_derivatives, prob.N_operators)
        control_vals_imag = zeros(1+N_derivatives, prob.N_operators)
        tnext=NaN
        new{typeof(prob)}(
            N_derivatives, dt, uv_mat, control_vals_real,
            control_vals_imag, prob
        )
    end
end

"""
Callable struct
"""
function (self::LHSHolder)(out_vec, in_vec)
    self.uv_mat[:,1] .= in_vec
    compute_derivatives!(
        self.uv_mat, self.prob, self.control_vals_real, self.control_vals_imag,
        self.N_derivatives
    )
    build_LHS!(out_vec, self.uv_mat, self.dt, self.N_derivatives)

    return nothing
end



struct LHSHolderAdjoint{T}
    N_derivatives::Int64
    dt::Float64
    uv_mat::Matrix{Float64}
    working_vec::Vector{Float64}
    working_matrix::Matrix{Float64}
    working_matrix2::Matrix{Float64}
    control_vals_real::Matrix{Float64}
    control_vals_imag::Matrix{Float64}
    prob::T
    function LHSHolderAdjoint(prob, N_derivatives, dt)
        uv_mat = zeros(prob.real_system_size, 1+N_derivatives)
        working_vec = zeros(prob.real_system_size)
        working_matrix = zeros(prob.real_system_size, 1+N_derivatives)
        working_matrix2 = zeros(prob.real_system_size, 1+N_derivatives)
        control_vals_real = zeros(1+N_derivatives, prob.N_operators)
        control_vals_imag = zeros(1+N_derivatives, prob.N_operators)
        tnext=NaN
        new{typeof(prob)}(
            N_derivatives, dt, uv_mat, working_vec, working_matrix, working_matrix2,
            control_vals_real, control_vals_imag, prob
        )
    end
end

"""
Work in progress, callable struct
"""
function (self::LHSHolderAdjoint)(out_vec, in_vec)
    self.uv_mat[:,1] .= in_vec
    compute_adjoint_derivatives!(
        self.uv_mat, self.prob, self.control_vals_real, self.control_vals_imag,
        self.N_derivatives, self.working_vec, self.working_matrix, self.working_matrix2
    )
    build_LHS!(out_vec, self.uv_mat, self.dt, self.N_derivatives)

    return nothing
end


function form_LHS_no_control(prob::SchrodingerProb, order::Int, adjoint=false)
    dt = prob.tf/prob.nsteps
    return form_LHS_no_control(prob.system_sym, prob.system_asym, order, dt, adjoint)
end

function form_LHS_no_control(system_sym::AbstractMatrix{Float64}, system_asym::AbstractMatrix{Float64}, order::Int, dt, adjoint=false)
    A = [system_asym system_sym; -system_sym system_asym]

    if adjoint
        # Without time dependence, all the matrices are A, A², A³, etc, so it's
        # okay to just take transpose directly
        A = A'
    end

    real_system_size = size(A, 1)
    complex_system_size = div(real_system_size, 2)
    N_derivatives = div(order, 2)

    LHS = similar(A) 
    LHS .= 0

    for i in 1:size(LHS, 2)
        LHS[i,i] = 1
    end
    for j in 1:N_derivatives
        coeff = (-dt)^j * coefficient(j, N_derivatives, N_derivatives)
        axpy!(coeff, A^j, LHS)
    end

    return LHS
end
