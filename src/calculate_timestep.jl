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
Should have a function that actually does an error analysis to figure out how
many timesteps are typically needed per period. Could randomize this over
many SchrodingerProbs and include it in the paper. Would be a good reference to
estimate how many timesteps would be needed for various problems.

# Arguments
- `prob::SchrodingerProb`
- `max_amplitudes::Vector{<: Real}`: Maximum amplitude for each control function.
- `timesteps_per_period::Integer`: Number of time steps per shortest period (assuming a slowly varying Hamiltonian).
"""
function estimate_timesteps_per_period(
        prob::SchrodingerProb,
        max_amplitudes::Vector{<: Real},
        order::Integer
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
    relative_errors = Any[]

    println("\n\nStarting Test with order $order\n\n")
    for (i, timesteps_per_period) in enumerate(timesteps_per_period_vec)
        prob.nsteps = estimate_N_timesteps(prob, max_amplitudes, timesteps_per_period)
        history = eval_forward(prob, controls, pcof, order=order)
        push!(histories, history)

        if i > 1
            relative_error = QuantumGateDesign.richardson_extrap_rel_err(
                histories[end][:,end,:], histories[end-1][:,end,:], order
            )
            push!(relative_errors, relative_error)
            println("For $(timesteps_per_period_vec[i]) timesteps per period, relative error is $relative_error")

            #if relative_error < 1e-9
            #    break
            #end
        end
    end
    
    return relative_errors
end
