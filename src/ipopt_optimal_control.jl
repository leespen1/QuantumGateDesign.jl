mutable struct OptimizationTracker
    last_pcof::Vector{Float64}
    last_grad_pcof::Vector{Float64}
    last_forward_evolution_pcof::Vector{Float64}
    last_discrete_adjoint_pcof::Vector{Float64}
    last_objective::Float64
    last_infidelity::Float64
    last_guard_penalty::Float64
    last_ridge_penalty::Float64
    last_length_deviation::Float64
    function OptimizationTracker(N_params::Integer)
        initial_pcof = fill(NaN, N_params)
        initial_grad_pcof = fill(NaN, N_params)
        initial_forward_evolution_pcof = fill(NaN, N_params)
        initial_discrete_adjoint_pcof = fill(NaN, N_params)

        new(initial_pcof, initial_grad_pcof, initial_forward_evolution_pcof, 
            initial_discrete_adjoint_pcof, NaN, NaN, NaN, NaN, NaN)
    end
end

struct OptimizationHistory
    ipopt_alg_mod::Vector{Int32}
    ipopt_iter::Vector{Int32}
    ipopt_objective::Vector{Float64}
    ipopt_inf_pr::Vector{Float64}
    ipopt_inf_du::Vector{Float64}
    ipopt_lg_mu::Vector{Float64}
    ipopt_d_norm::Vector{Float64}
    ipopt_regularization_size::Vector{Float64}
    ipopt_alpha_du::Vector{Float64}
    ipopt_alpha_pr::Vector{Float64}
    ipopt_ls::Vector{Int32}
    wall_time::Vector{Float64}
    pcof::Vector{Vector{Float64}}
    grad_pcof::Vector{Vector{Float64}}
    analytic_obj_value::Vector{Float64}
    infidelity::Vector{Float64}
    guard_penalty::Vector{Float64}
    ridge_penalty::Vector{Float64} # Add primaryobj, secondaryobj, to match juqbox
    length_deviation::Vector{Float64}
end

function OptimizationHistory()
    return OptimizationHistory(
        Int32[],
        Int32[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Int32[],
        Float64[],
        Vector{Float64}[],
        Vector{Float64}[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
    )
end

function Base.length(obj::OptimizationHistory)
    return length(obj.ipopt_iter)
end

function Base.show(io::IO, ::MIME"text/plain", obj::OptimizationHistory)
    println(io, typeof(obj))
    println(io, length(obj), " iterations performed.")

    if (length(obj) > 0)
        println(io, obj.wall_time[end], " seconds elapsed.")

        min_obj_val_index = argmin(obj.ipopt_objective)
        min_obj_val = obj.ipopt_objective[min_obj_val_index]
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
        file["ipopt_alg_mod"] = obj.ipopt_alg_mod
        file["ipopt_iter"] = obj.ipopt_iter
        file["ipopt_objective"] = obj.ipopt_objective
        file["ipopt_inf_pr"] = obj.ipopt_inf_pr
        file["ipopt_inf_du"] = obj.ipopt_inf_du
        file["ipopt_lg_mu"] = obj.ipopt_lg_mu
        file["ipopt_d_norm"] = obj.ipopt_d_norm
        file["ipopt_regularization_size"] = obj.ipopt_regularization_size
        file["ipopt_alpha_du"] = obj.ipopt_alpha_du
        file["ipopt_alpha_pr"] = obj.ipopt_alpha_pr
        file["ipopt_ls"] = obj.ipopt_ls
        file["wall_time"] = obj.wall_time
        file["pcof"] = obj.pcof
        file["grad_pcof"] = obj.grad_pcof
        file["analytic_obj_value"] = obj.analytic_obj_value
        file["infidelity"] = obj.infidelity
        file["guard_penalty"] = obj.guard_penalty
        file["ridge_penalty"] = obj.ridge_penalty
        file["length_deviation"] = obj.length_deviation
    end
end

"""
Write contents of an OptimizationHistory object (minus the vector fields) to a
csv
"""
function write_csv(obj::OptimizationTracker, filename)
end

"""
Read contents of a jld2 file into an OptimizationHistory object.
"""
function read_optimization_history(filename)
    jld2_dict = JLD2.load(filename)
    return OptimizationHistory(
        jld2_dict["ipopt_alg_mod"],
        jld2_dict["ipopt_iter"],
        jld2_dict["ipopt_objective"],
        jld2_dict["ipopt_inf_pr"],
        jld2_dict["ipopt_inf_du"],
        jld2_dict["ipopt_lg_mu"],
        jld2_dict["ipopt_d_norm"],
        jld2_dict["ipopt_regularization_size"],
        jld2_dict["ipopt_alpha_du"],
        jld2_dict["ipopt_alpha_pr"],
        jld2_dict["ipopt_ls"],
        jld2_dict["wall_time"],
        jld2_dict["pcof"],
        jld2_dict["grad_pcof"],
        jld2_dict["analytic_obj_value"],
        jld2_dict["infidelity"],
        jld2_dict["guard_penalty"],
        jld2_dict["ridge_penalty"],
        jld2_dict["length_deviation"]
    )
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
        pcof_init::AbstractVector{Float64}, target::AbstractMatrix{<: Number};
        order::Integer=4,
        pcof_lbound::Real=-Inf,
        pcof_ubound::Real=Inf,
        ridge_penalty_strength::Real=1e-2,
        savename::Union{Missing, String}=missing,
        ipopt_options=missing,
        write_every_iter=false,
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
    target_real_valued = vcat(real(target), imag(target))
    optimization_tracker = OptimizationTracker(N_coeff)
    optimization_history = OptimizationHistory()
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



    # Set up JLD2 file
    function write_jld2()
        if !ismissing(savename)
            jld2_filename = savename * ".jld2"
            JLD2.jldopen(jld2_filename, "w") do file
                # Also save SchrodingerProb, Controls, and Target, Optimization Parameters (one-time things that won't be updated)
                file["Setup/schrodinger_prob"] = schro_prob
                file["Setup/controls"] = controls
                file["Setup/target"] = target
                file["Setup/ridge_penalty_strength"] = ridge_penalty_strength
                file["Setup/pcof_init"] = pcof_init
                file["Setup/pcof_lbound"] = pcof_lbound
                file["Setup/pcof_ubound"] = pcof_ubound
                file["Setup/order"] = order
            end
            write(optimization_history, jld2_filename)
        end
    end

    write_jld2()

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

            # State vector length preservation - Check how much the state vector length deviates from unity
            length_deviation = 0.0
            for i in 1:size(QN, 2)
                ψf =  @view QN[:,i]
                ψf_length = LinearAlgebra.norm(ψf)
                ψf_length_deviation = ψf_length - 1
                length_deviation = abs(ψf_length_deviation) > abs(length_deviation) ? ψf_length_deviation : length_deviation
            end
            optimization_tracker.last_length_deviation = length_deviation
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

            # State vector length preservation - Check how much the state vector length deviates from unity
            length_deviation = 0.0
            for i in 1:size(QN, 2)
                ψf =  @view QN[:,i]
                ψf_length = LinearAlgebra.norm(ψf)
                ψf_length_deviation = ψf_length - 1
                length_deviation = abs(ψf_length_deviation) > abs(length_deviation) ? ψf_length_deviation : length_deviation
            end
            optimization_tracker.last_length_deviation = length_deviation
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
        push!(optimization_history.ipopt_alg_mod, alg_mod)
        push!(optimization_history.ipopt_iter, iter_count)
        push!(optimization_history.ipopt_objective, obj_value )
        push!(optimization_history.ipopt_inf_pr, inf_pr)
        push!(optimization_history.ipopt_inf_du, inf_du)
        push!(optimization_history.ipopt_lg_mu, mu) # The ipopt terminal output gives lg_mu. Is mu on log scale too?
        push!(optimization_history.ipopt_d_norm, d_norm)
        push!(optimization_history.ipopt_regularization_size, regularization_size)
        push!(optimization_history.ipopt_alpha_du, alpha_du)
        push!(optimization_history.ipopt_alpha_pr, alpha_pr)
        push!(optimization_history.ipopt_ls, ls_trials)
        push!(optimization_history.wall_time, elapsed_time)
        push!(optimization_history.pcof, optimization_tracker.last_pcof)
        push!(optimization_history.grad_pcof, optimization_tracker.last_grad_pcof)
        push!(optimization_history.analytic_obj_value, optimization_tracker.last_objective)
        push!(optimization_history.infidelity, optimization_tracker.last_infidelity)
        push!(optimization_history.guard_penalty, optimization_tracker.last_guard_penalty)
        push!(optimization_history.ridge_penalty, optimization_tracker.last_ridge_penalty)
        push!(optimization_history.length_deviation, optimization_tracker.last_length_deviation)

        # Open file in append mode and update arrays
        if write_every_iter
            write_jld2()
        end

        infidelity = optimization_tracker.last_infidelity

        ## Commenting this out so the log output is clean enough to parse
        #if (infidelity < 0) || (infidelity > 1)
        #    @warn "Infidelity $infidelity is outside range the [0,1]. This may indicate that the numerical error in the solution at the final time is greater than the deviation of the implemented gate from the target gate. Considert using a smaller stepsize."
        #end

        #if obj_value < 1e-7
        #    return false # Stop the optimization
        #end
        return true # continue the optimization
    end

    #==========================================================================
    # Define objective and gradient calculation, plus custom iteration callback
    ==========================================================================#

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

    # Save data
    write_jld2()

    return optimization_history
end

