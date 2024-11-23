mutable struct OptimizationTracker
    last_pcof::Vector{Float64}
    last_grad_pcof::Vector{Float64}
    last_forward_evolution_pcof::Vector{Float64}
    last_discrete_adjoint_pcof::Vector{Float64}
    last_objective::Float64
    last_infidelity::Float64
    last_guard_penalty::Float64
    last_ridge_penalty::Float64
    function OptimizationTracker(N_params::Integer)
        initial_pcof = fill(NaN, N_params)
        initial_grad_pcof = fill(NaN, N_params)
        initial_forward_evolution_pcof = fill(NaN, N_params)
        initial_discrete_adjoint_pcof = fill(NaN, N_params)

        new(initial_pcof, initial_grad_pcof, initial_forward_evolution_pcof, 
            initial_discrete_adjoint_pcof, NaN, NaN, NaN, NaN)
    end
end

struct OptimizationHistory
    iter_count::Vector{Int64}
    ipopt_obj_value::Vector{Float64}
    wall_time::Vector{Float64}
    pcof::Vector{Vector{Float64}}
    grad_pcof::Vector{Vector{Float64}}
    analytic_obj_value::Vector{Float64}
    infidelity::Vector{Float64}
    guard_penalty::Vector{Float64}
    ridge_penalty::Vector{Float64}
end

function OptimizationHistory()
    return OptimizationHistory(
        Int64[],
        Float64[],
        Float64[],
        Vector{Float64}[],
        Vector{Float64}[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
    )
end

function Base.length(obj::OptimizationHistory)
    return length(obj.iter_count)
end

function Base.show(io::IO, ::MIME"text/plain", obj::OptimizationHistory)
    println(io, typeof(obj))
    println(io, length(obj), " iterations performed.")

    if (length(obj) > 0)
        println(io, obj.wall_time[end], " seconds elapsed.")

        min_obj_val_index = argmin(obj.ipopt_obj_value)
        min_obj_val = obj.ipopt_obj_value[min_obj_val_index]
        println(io, "Minimum objective function was ", min_obj_val, ", at iteration ", min_obj_val_index, ".")

        min_infidelity_index = argmin(obj.infidelity)
        min_infidelity = obj.infidelity[min_infidelity_index]
        println(io, "Minimum infidelity was ", min_infidelity, ", at iteration ", min_infidelity_index, ".")
    end

    return nothing
end


"""
Write contents of an OptimizationHistory object to a jld2 file.
"""
function write(obj::OptimizationHistory, filename)
    JLD2.jldopen(filename, "a+") do file
        file["iter_count"] = obj.iter_count
        file["ipopt_obj_value"] = obj.ipopt_obj_value
        file["wall_time"] = obj.wall_time
        file["pcof"] = obj.pcof
        file["grad_pcof"] = obj.grad_pcof
        file["analytic_obj_value"] = obj.analytic_obj_value
        file["infidelity"] = obj.infidelity
        file["guard_penalty"] = obj.guard_penalty
        file["ridge_penalty"] = obj.ridge_penalty
    end
end

"""
Read contents of a jld2 file into an OptimizationHistory object.
"""
function read_optimization_history(filename)
    jld2_dict = JLD2.load(filename)
    return OptimizationHistory(
        jld2_dict["iter_count"],
        jld2_dict["ipopt_obj_value"],
        jld2_dict["wall_time"],
        jld2_dict["pcof"],
        jld2_dict["grad_pcof"],
        jld2_dict["analytic_obj_value"],
        jld2_dict["infidelity"],
        jld2_dict["guard_penalty"],
        jld2_dict["ridge_penalty"],
    )
end

# Wrappers for easy parameter adding
function genericAddOption(ipopt_prob, option, value::String)
    Ipopt.AddIpoptStrOption(ipopt_prob, option, value)
end

function genericAddOption(ipopt_prob, option, value::Integer)
    Ipopt.AddIpoptIntOption(ipopt_prob, option, value)
end

function genericAddOption(ipopt_prob, option, value::Number)
    Ipopt.AddIpoptNumOption(ipopt_prob, option, value)
end

# First set up constraint, jacobian, and hessian functions. We will not be
# using those in our optimization process (yet), so they all just return
# nothing.
#
# I have added exclamation points because their purpose really is to mutate
# vectors. They don't return anything meaningful.

"""
Unused, but need a function to provide to ipopt.
"""
function dummy_eval_g!(x::Vector{Float64}, g::Vector{Float64})
    return
end

"""
Unused, but need a function to provide to ipopt.
"""
function dummy_eval_jacobian_g!(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    values::Union{Nothing,Vector{Float64}},
)
    return
end

"""
Unused, but need a function to provide to ipopt.
"""
function dummy_eval_hessian!(
        x::Vector{Float64},
        rows::Vector{Int32},
        cols::Vector{Int32},
        obj_factor::Float64,
        lambda::Vector{Float64},
        values::Union{Nothing,Vector{Float64}},
    )

    return 
end


"""
    optimize_gate(schro_prob, controls, pcof_init, target, [order=4, pcof_L=missing, pcof_U=missing, maxIter=50, print_level=5, ridge_penalty_strength=1e-2, max_cpu_time = 300.0])

Perform gradient-based search (L-BFGS) to find value of the control vector `pcof`
which minimizes the objective function for the given problem and target.
Returns a dictionary which contains the ipopt optimization problem object, as
well as other information about the optimization.

NOTE: to play around with IPOPT settings which are not accessible through this
function call, could run the optimization with maxIter=1, then grab the IPOPT
problem from the return dictionary, and change the IPOPT settings directly through
the IPOPT API.

# Arguments
- `prob::SchrodingerProb`: Object containing the Hamiltonians, number of timesteps, etc.
- `controls`: An `AstractControl` or vector of controls, where the i-th control corresponds to the i-th control Hamiltonian.
- `pcof::AbstractVector{<: Real}`: The control vector.
- `target::AbstractMatrix{Float64}`: The target gate, in 'stacked' real-valued format.
- `order::Int64=2`: Which order of the timestepping method to use.
- `pcof_L=missing`: Lower bounds of the control parameters. Can either be a single number, used for all parameters, or a vector the same length as `pcof`, which will set a lower limit on each parameter.
- `pcof_U=missing`: Upper bounds of the control parameters.
- `maxIter=50`: Maximum number of iterations to perform.
- `print_level=5`: Print level of IPOPT.
- `ridge_penalty_strength`: Strength of the ridge/Tikhonov regularization term in the objective function.
- `max_cpu_time`: Maximum CPU time (in seconds) to spend on the optimization problem.
"""
function optimize_gate(
        schro_prob::SchrodingerProb{M, VM}, controls,
        pcof_init::AbstractVector{Float64}, target::AbstractMatrix{<: Number};
        order=4,
        pcof_L=missing,
        pcof_U=missing,
        maxIter=50,
        print_level=5, # Default is 5, goes from 0 to 12
        ridge_penalty_strength=1e-2,
        max_cpu_time = 60.0*60*24, # 24 hours
        filename=missing,
    ) where {VM<:AbstractVecOrMat{Float64}, M<:AbstractMatrix{Float64}}

    N_coeff = get_number_of_control_parameters(controls)
    @assert length(pcof_init) == N_coeff
    optimization_tracker = OptimizationTracker(N_coeff)
    optimization_history = OptimizationHistory()



    N_derivatives = div(order, 2)
    # Pre-allocate arrays
    state_history =  zeros(
        schro_prob.real_system_size,
        1+N_derivatives,
        1+schro_prob.nsteps,
        schro_prob.N_initial_conditions
    )
    lambda_history = similar(state_history)
    adjoint_forcing = zeros(schro_prob.real_system_size, 1+schro_prob.nsteps, schro_prob.N_initial_conditions)

    target_real_valued = vcat(real(target), imag(target))

    initial_time = NaN # Will overwrite this just before starting the actual optimization

    # Set up JLD2 file
    function update_jld2()
        if !ismissing(filename)
            JLD2.jldopen(filename, "w") do file
                # Also save SchrodingerProb, Controls, and Target, Optimization Parameters (one-time things that won't be updated)
                file["Setup/schrodinger_prob"] = schro_prob
                file["Setup/controls"] = controls
                file["Setup/target"] = target
                file["Setup/ridge_penalty_strength"] = ridge_penalty_strength
                file["Setup/max_cpu_time"] = max_cpu_time
                file["Setup/pcof_init"] = pcof_init
                file["Setup/pcof_L"] = pcof_L
                file["Setup/pcof_U"] = pcof_U
                file["Setup/order"] = order
            end
            write(optimization_history, filename)
        end
    end

    update_jld2()

    function eval_f(pcof::Vector{Float64})

        ## Check if control vector differs from old one before performing computation (maybe use relative error here?)
        ## My assumption is that we may compute the objective function many times without computing the gradient,
        ## and that whenever we compute the gradient we will also want the objective function

        #pcof_difference = LinearAlgebra.norm(pcof - optimization_tracker.last_pcof)
        #if (pcof_difference > 1e-15) || !isfinite(pcof_difference)

        # If pcof has changed, need to recalculate objective function
        # (if it stayed the same but eval_grad_f! was called before eval_f,
        # don't need to do anything since eval_grad_f! also computes the
        # objective function)
        if (pcof != optimization_tracker.last_pcof)
            eval_forward!(state_history, schro_prob, controls, pcof, order=order)
            QN = @view state_history[:,1,end,:]
            # Infidelity Term
            optimization_tracker.last_infidelity = infidelity_real(
                QN, target_real_valued, schro_prob.N_ess_levels
            ) 

            # Guard Penalty Term
            dt = schro_prob.tf / schro_prob.nsteps

            optimization_tracker.last_guard_penalty = guard_penalty_real(
                state_history, dt, schro_prob.tf, schro_prob.guard_subspace_projector
            )

            # Ridge/L2 Penalty Term
            ridge_pen_val = dot(pcof, pcof)*ridge_penalty_strength / length(pcof)
            optimization_tracker.last_ridge_penalty = ridge_pen_val

            # Add all objective terms
            optimization_tracker.last_objective = sum((
                optimization_tracker.last_infidelity,
                optimization_tracker.last_guard_penalty,
                optimization_tracker.last_ridge_penalty,
            ))
            optimization_tracker.last_pcof .= pcof
            optimization_tracker.last_forward_evolution_pcof .= pcof
        end

        return optimization_tracker.last_objective
    end

    
    function eval_grad_f!(pcof::Vector{Float64}, grad_f::Vector{Float64})
        
        ## Should I check equality or just for small differences?
        #pcof_difference = LinearAlgebra.norm(pcof - optimization_tracker.last_pcof)
        #if (pcof_difference > 1e-15) || !isfinite(pcof_difference)  || !optimization_tracker.adjoint_calculated

        ## Cover case where pcof changes and we immediate eval_grad_f!, and case
        ## where we run eval_f, don't change pcof, and then run eval_grad_f!
        if (pcof != optimization_tracker.last_discrete_adjoint_pcof)

            # If we already ran the objective evaluation, then we can reuse the state history from the forward evolution 
            # (but we may also have run eval_grad_f! for a brand new pcof, so I am being careful of that)
            history_precomputed = (pcof == optimization_tracker.last_forward_evolution_pcof)
            println("history_precomputed = ", history_precomputed)

            discrete_adjoint!(
                optimization_tracker.last_grad_pcof, state_history,
                lambda_history, adjoint_forcing, schro_prob, controls, pcof,
                target, order=order, history_precomputed=history_precomputed
            )
            # Ridge Regression Penalty (not included in main discrete adjoint, not necessary since nothing depends on the states)
            N_coeff = length(pcof)
            @. optimization_tracker.last_grad_pcof += 2.0*ridge_penalty_strength*pcof / N_coeff

            optimization_tracker.last_pcof .= pcof
            optimization_tracker.last_discrete_adjoint_pcof .= pcof

            #
            # Also calculate objective function, just because it's not expensive, it helps with eval_f
            #
            QN = @view state_history[:,1,end,:]
            # Infidelity Term
            optimization_tracker.last_infidelity = infidelity_real(
                QN, target_real_valued, schro_prob.N_ess_levels
            ) 

            # Guard Penalty Term
            dt = schro_prob.tf / schro_prob.nsteps

            optimization_tracker.last_guard_penalty = guard_penalty_real(
                state_history, dt, schro_prob.tf, schro_prob.guard_subspace_projector
            )

            # Ridge/L2 Penalty Term
            ridge_pen_val = dot(pcof, pcof)*ridge_penalty_strength / length(pcof)
            optimization_tracker.last_ridge_penalty = ridge_pen_val

            # Add all objective terms
            optimization_tracker.last_objective = sum((
                optimization_tracker.last_infidelity,
                optimization_tracker.last_guard_penalty,
                optimization_tracker.last_ridge_penalty,
            ))
        end

        grad_f .= optimization_tracker.last_grad_pcof
        return nothing
    end

    function my_callback(
        alg_mod,
        iter_count,
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        ls_trials
    )
        elapsed_time = time() - initial_time
        # Open file in append mode and update arrays
        push!(optimization_history.iter_count, iter_count)
        push!(optimization_history.ipopt_obj_value, obj_value)
        push!(optimization_history.wall_time, elapsed_time)
        push!(optimization_history.pcof, optimization_tracker.last_pcof)
        push!(optimization_history.grad_pcof, optimization_tracker.last_grad_pcof)
        push!(optimization_history.analytic_obj_value, optimization_tracker.last_objective)
        push!(optimization_history.infidelity, optimization_tracker.last_infidelity)
        push!(optimization_history.guard_penalty, optimization_tracker.last_guard_penalty)
        push!(optimization_history.ridge_penalty, optimization_tracker.last_ridge_penalty)

        update_jld2()


        infidelity = optimization_tracker.last_infidelity
        if (infidelity < 0) || (infidelity > 1)
            @warn "Infidelity is outside range the [0,1]. This may indicate that the solution is inaccurate, and a smaller stepsize is needed."
        end

        if obj_value < 1e-7
            return false # Stop the optimization
        end
        return true # continue the optimization
    end

    N_parameters = length(pcof_init)

    if ismissing(pcof_L)
        # If lower bounds not provided, set lower bound to -Inf  
        pcof_L = ones(N_parameters)*-Inf
    elseif isa(pcof_L, Real)
        # If one value provided, apply that bound to all parameters
        pcof_L = ones(N_parameters) .* pcof_L
    end

    if ismissing(pcof_U)
        # If upper bounds not provided, set upper bound to Inf
        pcof_U = ones(N_parameters)*Inf
    elseif isa(pcof_U, Real)
        # If one value provided, apply that bound to all parameters
        pcof_U = ones(N_parameters) .* pcof_U
    end

    @assert isa(pcof_L, Vector{Float64})
    @assert isa(pcof_U, Vector{Float64})

    N_constraints = 0
    g_L = Vector{Float64}()
    g_U = Vector{Float64}()

    nele_jacobian = 0
    nele_hessian = 0

    ipopt_prob = Ipopt.CreateIpoptProblem(
        N_parameters,
        pcof_L,
        pcof_U,
        N_constraints,
        g_L,
        g_U,
        nele_jacobian,
        nele_hessian,
        eval_f,
        dummy_eval_g!,
        eval_grad_f!,
        dummy_eval_jacobian_g!,
        dummy_eval_hessian!,
    )

    lbfgsMax = 40
    acceptTol = 5e-5 
    ipTol = 1e-5
    acceptIter = 15 # Number of "acceptable" iterations before calling it quits
    

    # Should add derivative test back in. I think this tests for correct
    # derivatives? Maybe gradients?

    # Description of options: https://coin-or.github.io/Ipopt/OPTIONS.html

    Ipopt.AddIpoptStrOption(ipopt_prob, "hessian_approximation", "limited-memory"); # Use L-BFGS, approximate hessian
    Ipopt.AddIpoptIntOption(ipopt_prob, "limited_memory_max_history", lbfgsMax); # Maximum number of gradients to use for Hessian approximation (not really a memory concern for me)
    Ipopt.AddIpoptIntOption(ipopt_prob, "max_iter", maxIter); # Maximum number of iterations to run before terminating
    Ipopt.AddIpoptNumOption(ipopt_prob, "max_cpu_time", max_cpu_time); # Maximum number of iterations to run before terminating
    Ipopt.AddIpoptNumOption(ipopt_prob, "tol", ipTol); # Relative convergence tolerance. Terminate if (scaled) NLP error becomes smaller than this (NLP = Nonlinear Programming, is NLP error just objective function, or something closely related?)
    #Ipopt.AddIpoptNumOption(ipopt_prob, "acceptable_tol", acceptTol); # "Acceptable" relative convergence tolerance
    Ipopt.AddIpoptIntOption(ipopt_prob, "acceptable_iter", acceptIter); # If we perform this many iterations with "acceptable" NLP error, terminate optimization process (useful if we can't reach desired tolerance)
    Ipopt.AddIpoptStrOption(ipopt_prob, "jacobian_approximation", "exact");
    #Ipopt.AddIpoptStrOption(ipopt_prob, "derivative_test", "first-order") # What does this do?
    #Ipopt.AddIpoptStrOption(ipopt_prob, "derivative_test", "none") # Not sure what derivative test does, but it takes a minute.
    Ipopt.AddIpoptIntOption(ipopt_prob, "print_level", print_level)  
    #Ipopt.AddIpoptStrOption(ipopt_prob, "timing_statistics", "yes")
    #Ipopt.AddIpoptStrOption(ipopt_prob, "print_timing_statistics", "yes")
    # Anything below this number will be considered -∞. I.e., terminate when objective goes beloww this
    #Ipopt.AddIpoptNumOption(ipopt_prob, "nlp_lower_bound_inf", nlp_lower_bound)
    
    # Trying to figure out why my ls is frequently bigger than in juqbox
    # If I add this, the optimization suffers greatly.
    #Ipopt.AddIpoptStrOption(ipopt_prob, "accept_every_trial_step", "yes")
    Ipopt.SetIntermediateCallback(ipopt_prob, my_callback)


    ipopt_prob.x .= pcof_init

    initial_time = time()

    solvestat = Ipopt.IpoptSolve(ipopt_prob)

    return optimization_history
end

