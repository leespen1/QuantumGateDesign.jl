"""
    nsteps = estimate_N_timesteps(prob, max_amplitudes, [timesteps_per_period=40])

Estimate the number of timesteps needed for the simulation based on the number
of timesteps desired per shortest period.

Does not account for the dynamics within the control functions, and numerical
tests should be performed to verify the accuracy when using an actual control
function. E.g. use richardson extrapolation to estimate the error in the
solution.
 
# Arguments
- `prob::SchrodingerProb`
- `max_amplitudes::Vector{<: Real}`: Maximum amplitude for each control function.
- `timesteps_per_period::Integer`: Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function get_shortest_period(prob::SchrodingerProb, max_amplitudes::Vector{<: Real})

    full_hamiltonian = prob.system_sym + 1im*prob.system_asym
    for i in 1:prob.N_operators
        full_hamiltonian += max_amplitudes[i] * prob.sym_operators[i]
        full_hamiltonian += 1im * max_amplitudes[i] * prob.asym_operators[i]
    end

    # Estimate time step
    eigenvalues = LinearAlgebra.eigvals(Matrix(full_hamiltonian)) # eigvals expects plain Matrix type

    max_eig = maximum(abs.(eigenvalues)) 

    shortest_period = 2pi / max_eig

    return shortest_period
end

function estimate_N_timesteps(
        prob::SchrodingerProb,
        max_amplitudes::Vector{<: Real},
        timesteps_per_period::Real=40
    )
    shortest_period = get_shortest_period(prob, max_amplitudes)
    number_of_periods = prob.tf / shortest_period

    nsteps = ceil(Int64, number_of_periods * timesteps_per_period)
    return nsteps
end

"""
Using the max-amplitude strategy, perform actual forward evolutions to determine
an appropiate number timesteps per period to achieve the target error.

# Arguments
- `prob::SchrodingerProb`
- `max_amplitudes::Vector{<: Real}`: Maximum amplitude for each control function.
- `timesteps_per_period::Integer`: Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function experiment_N_timesteps(
        prob::SchrodingerProb,
        max_amplitudes::Vector{<: Real},
        order::Integer,
        target_error::Real
    )
    # We are going to mutate the stepsize, so copy the problem
    prob = copy(prob)

    # Use the piecewise-constant GRAPE control to make constant controls at max amplitude
    N_amplitudes = 1
    controls = [GRAPEControl(N_amplitudes, prob.tf) for i in 1:prob.N_operators]
    # Repeat each max amplitude twice, once for the real control and once for the imaginary control
    pcof = repeat(max_amplitudes, inner=2)


    timesteps_per_period_vec = [2.0^i for i in -3:6]
    histories = Any[] 
    relative_errors = Float64[]
    nsteps_vec = Float64[]

    println("\n\nStarting Test with order $order\n\n")
    for (i, timesteps_per_period) in enumerate(timesteps_per_period_vec)
        nsteps = estimate_N_timesteps(prob, max_amplitudes, timesteps_per_period)
        prob.nsteps = nsteps
        history = eval_forward(prob, controls, pcof, order=order)
        push!(histories, history)
        push!(nsteps_vec, nsteps)

        if i > 1
            relative_error = QuantumGateDesign.richardson_extrap_rel_err(
                histories[end][:,end,:], histories[end-1][:,end,:], order
            )
            push!(relative_errors, relative_error)
            println("For $(timesteps_per_period_vec[i]) timesteps per period, relative error is $relative_error")

            # Stop once we have gotten below the target error
            if (relative_error < target_error) && (i > 2)
                break
            end
        end
    end
    push!(relative_errors, NaN)

    # Do a linear interpolation of the last two relative errors (in log scale)
    # to estimate the stepsize needed to achieve the target error

    log10_rel_errs = log10.(relative_errors)
    log10_nsteps = log10.(nsteps_vec)

    log10_target_nsteps = linear_interpolate(
        log10_nsteps[end-2], log10_rel_errs[end-2],
        log10_nsteps[end-1], log10_rel_errs[end-1],
        log10(target_error)    
    )

    display(hcat(log10_nsteps, log10_rel_errs))
    println(log10_target_nsteps)

    target_nsteps = 10.0 ^ log10_target_nsteps
    
    return ceil(Int64, target_nsteps)
end

function linear_interpolate(x1, y1, x2, y2, y_target)
    x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
    return x_target
end
