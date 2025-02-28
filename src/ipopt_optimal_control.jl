mutable struct OptimizationTracker
    last_pcof::Vector{Float64}
    last_grad_pcof::Vector{Float64}
    last_forward_evolution_pcof::Vector{Float64}
    last_discrete_adjoint_pcof::Vector{Float64}
    last_objective::Float64
    last_main_objective::Float64
    last_infidelity::Float64
    last_generalized_infidelity::Float64
    last_tracking_obj::Float64
    last_norm_obj::Float64
    last_guard_penalty::Float64
    last_ridge_penalty::Float64
    last_avg_state_length::Float64
    function OptimizationTracker(N_params::Integer)
        initial_pcof = fill(NaN, N_params)
        initial_grad_pcof = fill(NaN, N_params)
        initial_forward_evolution_pcof = fill(NaN, N_params)
        initial_discrete_adjoint_pcof = fill(NaN, N_params)

        new(initial_pcof, initial_grad_pcof, initial_forward_evolution_pcof, 
            initial_discrete_adjoint_pcof, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end
end

function update!(opt::OptimizationTracker, schro_prob::SchrodingerProb,
        state_history_real, pcof, target_complex, cost_type, ridge_penalty_strength
    )

    QN_complex = real_to_complex(state_history_real[:,1,end,:])

    opt.last_main_objective = cost_function(
        QN_complex, target_complex, schro_prob.N_ess_levels,
        cost_type=cost_type
    ) 

    # Infidelity Term
    opt.last_infidelity = cost_function(
        QN_complex, target_complex, schro_prob.N_ess_levels,
        cost_type=:Infidelity
    ) 

    opt.last_generalized_infidelity = cost_function(
        QN_complex, target_complex, schro_prob.N_ess_levels,
        cost_type=:GeneralizedInfidelity
    ) 

    opt.last_tracking_obj = cost_function(
        QN_complex, target_complex, schro_prob.N_ess_levels,
        cost_type=:Tracking
    ) 

    opt.last_norm_obj = cost_function(
        QN_complex, target_complex, schro_prob.N_ess_levels,
        cost_type=:Norm
    ) 

    # Guard Penalty Term
    dt = schro_prob.tf / schro_prob.nsteps
    opt.last_guard_penalty = guard_penalty_real(
        state_history_real, dt, schro_prob.tf, schro_prob.guard_subspace_projector
    )

    # Ridge/L2 Penalty Term
    ridge_pen_val = dot(pcof, pcof)*ridge_penalty_strength / length(pcof)
    opt.last_ridge_penalty = ridge_pen_val

    # Add all objective terms
    opt.last_objective = sum((
        opt.last_main_objective,
        opt.last_guard_penalty,
        opt.last_ridge_penalty,
    ))
    opt.last_pcof .= pcof

    # State vector length preservation - Check how much the state vector length deviates from unity
    total_length = 0.0
    for state in eachcol(QN_complex)
        state_length = LinearAlgebra.norm(state)
        total_length += state_length
    end
    opt.last_avg_state_length = total_length / size(QN_complex, 2)
end


# Wrappers for easy parameter adding
function AddIpoptOption(prob::Ipopt.IpoptProblem, keyword::String, value::String)
    Ipopt.AddIpoptStrOption(prob, keyword, value)
end

function AddIpoptOption(prob::Ipopt.IpoptProblem, keyword::String, value::Float64)
    Ipopt.AddIpoptNumOption(prob, keyword, value)
end

function AddIpoptOption(prob::Ipopt.IpoptProblem, keyword::String, value::Int64)
    try
        Ipopt.AddIpoptIntOption(prob, keyword, value)
    catch e
        println("Error while trying to set Ipopt integer option '$keyword' to '$value', trying again as Num option.")
        Ipopt.AddIpoptNumOption(prob, keyword, Float64(value))
    end
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
- `ridge_penalty_strength`: Strength of the ridge/Tikhonov regularization term in the objective function.
"""
function optimize_gate(
        schro_prob::SchrodingerProb{M, VM}, controls,
        pcof_init::AbstractVector{Float64}, target_complex::AbstractMatrix{<: Number};
        order::Integer=4,
        pcof_lbound::Real=-Inf,
        pcof_ubound::Real=Inf,
        ridge_penalty_strength::Real=1e-2,
        savename::Union{Missing, String}=missing,
        ipopt_options=missing,
        cost_type=:Infidelity,
    ) where {VM<:AbstractVecOrMat{Float64}, M<:AbstractMatrix{Float64}}


    # Check correct control vector length
    N_coeff = get_number_of_control_parameters(controls)
    if length(pcof_init) != N_coeff
        throw(ArgumentError("Length $(length(pcof_init)) of initial control vector does not match expected length based on the control functions ($N_coeff"))
    end

    # Set up variables neede to construct ipopt problem
    pcof_lbound_array = ones(N_coeff)*pcof_lbound
    pcof_ubound_array = ones(N_coeff)*pcof_ubound

    N_constraints = 0
    g_L = Float64[]
    g_U = Float64[]

    nele_jacobian = 0
    nele_hessian = 0

    # Other variables needed to interface with my code, also store information my way
    N_derivatives = div(order, 2)
    target_real_valued = vcat(real(target_complex), imag(target_complex))
    optimization_tracker = OptimizationTracker(N_coeff)
    initial_time = NaN # Will overwrite this just before starting the actual optimization

    
    # Pre-allocate arrays 
    state_history =  zeros(
        schro_prob.real_system_size,
        1+N_derivatives,
        1+schro_prob.nsteps,
        schro_prob.N_initial_conditions
    )
    lambda_history = similar(state_history)
    adjoint_forcing = zeros(schro_prob.real_system_size, 1+schro_prob.nsteps, schro_prob.N_initial_conditions)

    header = ["objective" "main_objective" "grad_norm" "infidelity" "generalized_infidelity" "tracking_objective" "norm_objective" "guard_penalty" "ridge_penalty" "avg_state_length" "elapsed_time" "alg_mod" "iter_count" "obj_value" "inf_pr" "inf_du" "mu" "d_norm" "regularization_size" "alpha_du" "alpha_pr" "ls_trials"]
    if !ismissing(savename)
        open(savename * ".csv", "w") do io
            DelimitedFiles.writedlm(io, header, ',')
        end
    end

    #==========================================================================
    # Define objective and gradient calculation, plus custom iteration callback
    ==========================================================================#
    
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
            update!(optimization_tracker, schro_prob, state_history, pcof, 
                    target_complex, cost_type, ridge_penalty_strength)
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
            #println("history_precomputed = ", history_precomputed)

            discrete_adjoint!(
                optimization_tracker.last_grad_pcof, state_history,
                lambda_history, adjoint_forcing, schro_prob, controls, pcof,
                target_complex, order=order, history_precomputed=history_precomputed,
                cost_type=cost_type,
            )
            # Ridge Regression Penalty (not included in main discrete adjoint, not necessary since nothing depends on the states)
            N_coeff = length(pcof)
            @. optimization_tracker.last_grad_pcof += 2.0*ridge_penalty_strength*pcof / N_coeff
            update!(optimization_tracker, schro_prob, state_history, pcof, 
                    target_complex, cost_type, ridge_penalty_strength)
            optimization_tracker.last_discrete_adjoint_pcof .= pcof
        end

        grad_f .= optimization_tracker.last_grad_pcof

        return nothing
    end

    function my_callback(
        alg_mod, # algorithm mode
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
        grad_norm = norm(optimization_tracker.last_grad_pcof)
        data_row = [optimization_tracker.last_objective optimization_tracker.last_main_objective grad_norm optimization_tracker.last_infidelity optimization_tracker.last_generalized_infidelity optimization_tracker.last_tracking_obj optimization_tracker.last_norm_obj optimization_tracker.last_guard_penalty optimization_tracker.last_ridge_penalty optimization_tracker.last_avg_state_length elapsed_time alg_mod iter_count obj_value inf_pr inf_du mu d_norm regularization_size alpha_du alpha_pr ls_trials]
        if !ismissing(savename)
            open(savename * ".csv", "a+") do io
                DelimitedFiles.writedlm(io, data_row, ',')
            end

            open(savename * "_pcof.csv", "a+") do io
                DelimitedFiles.writedlm(io, reshape(optimization_tracker.last_pcof, 1, :), ',')
            end
            open(savename * "_gradPcof.csv", "a+") do io
                DelimitedFiles.writedlm(io, reshape(optimization_tracker.last_grad_pcof, 1, :), ',')
            end
        end

        # Could put a stopping condition here if I want to

        return true # continue the optimization
    end

    ipopt_prob = Ipopt.CreateIpoptProblem(
        N_coeff,
        pcof_lbound_array,
        pcof_ubound_array,
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

    Ipopt.SetIntermediateCallback(ipopt_prob, my_callback)
    

    # Set default ipopt options and add them to the Ipopt problem
    # Description of options: https://coin-or.github.io/Ipopt/OPTIONS.html
    default_ipopt_options = (
        "hessian_approximation" => "limited-memory",
        "limited_memory_max_history" => 40,
        "max_iter" => 50,
        "acceptable_iter" => 15, # Number of "acceptable" iterations before calling it quits
        "tol" => 1e-5,
        "print_level" => 5, # Default is 5, goes from 0 to 12
        "derivative_test" => "none", # Change t "first-order" to do finite-difference check of derivatives
        "jacobian_approximation" => "exact", # I don't think this matters, since we don't compute the jacobian
        "max_cpu_time" => 60.0*60*24, # Default to 24 hours
    )

    for (keyword, value) in default_ipopt_options
        AddIpoptOption(ipopt_prob, keyword, value)
    end

    # Add user ipopt options, overriding defaults
    if !ismissing(ipopt_options)
        for (keyword, value) in ipopt_options
            AddIpoptOption(ipopt_prob, keyword, value)
        end
    end

    # Initialize
    ipopt_prob.x .= pcof_init
    initial_time = time()

    # Perform the optimization
    solvestat = Ipopt.IpoptSolve(ipopt_prob)

    return ipopt_prob
end

