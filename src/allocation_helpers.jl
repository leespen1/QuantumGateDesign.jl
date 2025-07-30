"""
Allocate 3D array to store state vector history.
Indices access: state index, derivative order, timestep.

TODO: Can I merge this with the matrix/4D version and still have it be type-stable?
"""
function allocate_history(prob::SchrodingerProb{OpType, StateType, P},
        method_order::Integer, nsteps_override::Union{Integer, Nothing}=nothing) where
        {OpType, StateType <: AbstractVector, P}

    @assert iseven(method_order)
    N_derivatives = div(method_order, 2)
    nsteps = isnothing(nsteps_override) ? prob.nsteps : nsteps_override

    history_alloc = zeros(
        eltype(StateType),
        prob.real_system_size,
        1+N_derivatives,
        1+nsteps,
    )
    return history_alloc
end

"""
Allocate 4D array to store state matrix history.
Indices access: stateindex, derivative order, timestep, initial condition index.
"""
function allocate_history(prob::SchrodingerProb{OpType, StateType, P},
        method_order::Integer, nsteps_override::Union{Integer, Nothing}=nothing) where
        {OpType, StateType <: AbstractMatrix, P}
    @assert iseven(method_order)
    N_derivatives = div(method_order, 2)
    nsteps = isnothing(nsteps_override) ? prob.nsteps : nsteps_override

    history_alloc = zeros(
        eltype(StateType),
        prob.real_system_size,
        1+N_derivatives,
        1+nsteps,
        prob.N_initial_conditions,
    )
    return history_alloc
end

"""
Allocate 2D array to store forcing history.
Indices access: state index, timestep index.
"""
function allocate_forcing(prob::SchrodingerProb{OpType, StateType, P},
        method_order::Integer) where
        {OpType, StateType <: AbstractVector, P}

    @assert iseven(method_order)
    N_derivatives = div(method_order, 2)

    history_alloc = zeros(
        eltype(StateType),
        prob.real_system_size,
        1+prob.nsteps,
    )
    return history_alloc
end

"""
Allocate 3D array to store state matrix history.
Indices access: state index, timestep index, initial condition index.
"""
function allocate_forcing(prob::SchrodingerProb{OpType, StateType, P},
        method_order::Integer) where
        {OpType, StateType <: AbstractMatrix, P}
    @assert iseven(method_order)
    N_derivatives = div(method_order, 2)

    history_alloc = zeros(
        eltype(StateType),
        prob.real_system_size,
        1+prob.nsteps,
        prob.N_initial_conditions,
    )
    return history_alloc
end
