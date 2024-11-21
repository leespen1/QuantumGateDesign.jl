import Ipopt

mutable struct OptimizationTracker
    last_pcof::Vector{Float64}
    last_objective::Float64
    last_infidelity::Float64
    last_guard_penalty::Float64
    last_ridge_penalty::Float64
    adjoint_calculated::Bool # Whether the adjoint has been calculated (if not, only forward evolution has been calculated)
    function ObjectiveTracking(N_params::Integer)
        initial_pcof = fill(NaN, N_params)
        new(initial_pcof, NaN, NaN, NaN, NaN, true)
    end
end

# Wrappers for easy parameter adding
function genericAddOption(ipopt_prob, option, value::String)
    Ipopt.AddIpoptStrOption(ipopt_prob, option, value)
end

function genericAddOption(ipopt_prob, option, value::Int)
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
        max_cpu_time = 60.0*60*24 # 24 hours
    ) where {VM<:AbstractVecOrMat{Float64}, M<:AbstractMatrix{Float64}}

    N_coeff = 1
    optimization_tracker = OptimizationTracker(1)

    pcof_history = []
    pcof_grad_history = []
    full_objective_history = []
    iter_objective_history = []
    iter_cpu_time_history = []

    N_derivatives = div(order, 2)
    state_history =  zeros(
        schro_prob.real_system_size,
        1+N_derivatives,
        1+schro_prob.nsteps,
        schro_prob.N_initial_conditions
    )

    target_real_valued = vcat(real(target), imag(target))




    # Right now I am unnecessarily doing a full forward evolution to compute the
    # infidelity, when this should be done already in the gradient computation.
    function eval_f(pcof::Vector{Float64})

        # Check if control vector differs from old one before performing computation (maybe use relative error here?)
        # My assumption is that we may compute the objective function many times without computing the gradient,
        # and that whenever we compute the gradient we will also want the objective function
        if LinearAlgebra.norm(pcof - optimization_tracker.last_pcof) > 1e-15
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
            optimization_tracker.adjoint_calculated = false
        end

        return optimization_tracker.last_objective
    end

    
    function eval_grad_f!(pcof::Vector{Float64}, grad_f::Vector{Float64})
        
        # Only update gradient if pcof has changed, or if pcof is the same but last time we only computed the objective function
        # Could make this more efficient by preallocating memory for the lambda history and gradient,
        # and also by reusing the forward evolution state from the objective function
        # DO BOTH THESE THINGS!
        if (LinearAlgebra.norm(pcof - optimization_tracker.last_pcof) > 1e-15) || !optimization_tracker.adjoint_calculated
            grad_f .= discrete_adjoint(schro_prob, controls, pcof, target, order=order)

            # Ridge Regression Penalty (not included in main discrete adjoint, not necessary since nothing depends on the states)
            N_coeff = length(pcof)
            @. grad_f += 2.0*ridge_penalty_strength*pcof / N_coeff
            optimization_tracker.last_pcof .= pcof
            optimization_tracker.adjoint_calculated = true
        end

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

        # Open file in append mode and update arrays
        jldopen("optimization_log.jld2", "a+") do file
            # Create entries if they don't already exist
            if !haskey(file, "iterations")
                file["iter_count"] = Int64[]
                file["ipopt_obj_value"] = Float64[]
                file["wall_time"] = Float64[]
                file["pcof"] = Vector{Float64}[]
                file["analytic_obj_value"] = Float64[]
                file["infidelity"] = Float64[]
                file["guard_penalty"] = Float64[]
                file["ridge_penalty"] = Float64[]
            end
            # Push new values to each entry
            push!(file["iter_count"], iter_count)
            push!(file["ipopt_obj_value"], obj_value)
            push!(file["wall_time"], time())
            push!(file["pcof"], optimization_tracker.last_pcof)
            push!(file["analytic_obj_value"], optimization_tracker.last_objective)
            push!(file["infidelity"], optimization_tracker.last_infidelity)
            push!(file["guard_penalty"], optimization_tracker.last_guard_penalty)
            push!(file["ridge_penalty"], optimization_tracker.last_guard_penalty)
        end

        return true  # continue the optimization (could have an infidelity cutoff here)
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
    solvestat = Ipopt.IpoptSolve(ipopt_prob)
    # solvestat I think is stored in ipopt_prob as 'status'
    
    # Go from system time since epoch to system time since iteration 0.
    @. iter_cpu_time_history -= iter_cpu_time_history[1]

    return_dict = OrderedCollections.OrderedDict(
        "ipopt_prob" => ipopt_prob,
        "final_objective_value" => ipopt_prob.obj_val,
        "optimal_pcof" => copy(ipopt_prob.x),
        "full_objective_history" => full_objective_history,
        "pcof_history" => pcof_history,
        "pcof_grad_history" => pcof_grad_history,
        "iter_objective_history" => iter_objective_history,
        "iter_cpu_time_history" => iter_cpu_time_history,
    )

    return return_dict
end

