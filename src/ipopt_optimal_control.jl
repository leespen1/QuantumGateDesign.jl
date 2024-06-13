import Ipopt

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
Unused, but need a function to provide to ipopt.
"""
function optimize_gate(
        schro_prob::SchrodingerProb{M, VM}, controls,
        pcof_init::AbstractVector{Float64}, target::VM;
        order=4,
        pcof_L=missing,
        pcof_U=missing,
        maxIter=50,
        print_level=5, # Default is 5, goes from 0 to 12
        ridge_penalty_strength=1e-2,
        max_cpu_time = 300.0 # 5 minutes
    ) where {VM<:AbstractVecOrMat{Float64}, M<:AbstractMatrix{Float64}}


    pcof_history = []
    pcof_grad_history = []
    full_objective_history = []
    iter_objective_history = []
    iter_cpu_time_history = []


    # Right now I am unnecessarily doing a full forward evolution to compute the
    # infidelity, when this should be done already in the gradient computation.
    function eval_f(pcof::Vector{Float64})
        history = eval_forward(schro_prob, controls, pcof, order=order)

        # Infidelity Term
        QN = @view history[:,1,end,:]
        infidelity_val = infidelity(QN, target, schro_prob.N_ess_levels) 

        # Guard Penalty Term
        dt = schro_prob.tf / schro_prob.nsteps
        guard_pen_val = guard_penalty(history, dt, schro_prob.tf, schro_prob.guard_subspace_projector)

        # Ridge/L2 Penalty Term
        ridge_pen_val = dot(pcof, pcof)*ridge_penalty_strength / length(pcof)
        
        objective_val = infidelity_val + guard_pen_val + ridge_pen_val

        push!(full_objective_history, objective_val)

        return objective_val
    end

    function eval_grad_f!(pcof::Vector{Float64}, grad_f::Vector{Float64})
        grad_f .= discrete_adjoint(schro_prob, controls, pcof, target, order=order)

        # Ridge Regression Penalty
        N_coeff = length(pcof)
        @. grad_f += 2.0*ridge_penalty_strength*pcof / N_coeff

        push!(pcof_history, copy(pcof))
        push!(pcof_grad_history, copy(grad_f))
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

        push!(iter_objective_history, obj_value)
        push!(iter_cpu_time_history, time())
        return true  # continue the optimization
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
    Ipopt.AddIpoptStrOption(ipopt_prob, "timing_statistics", "yes")
    Ipopt.AddIpoptStrOption(ipopt_prob, "print_timing_statistics", "yes")
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
        "full_objective_history" => full_objective_history,
        "pcof_history" => pcof_history,
        "pcof_grad_history" => pcof_grad_history,
        "iter_objective_history" => iter_objective_history,
        "iter_cpu_time_history" => iter_cpu_time_history,
    )

    return return_dict
end

