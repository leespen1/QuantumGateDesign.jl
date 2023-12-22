"""
Make a 1D problem with no controls. Corresponds to the dahlquist equation

y' = λy
"""
function dahlquist_problem(lambda::Number; initial_condition::Number=1.0, with_control=false)

    N_ess_levels = 1
    N_guard_levels = 0

    tf = 1.0
    nsteps = 10

    u0::Vector{Float64} = [real(initial_condition)]
    v0::Vector{Float64} = [imag(initial_condition)]

    system_sym = Matrix{Float64}(undef, 1, 1)
    system_asym = Matrix{Float64}(undef, 1, 1)

    system_sym .= real(lambda)
    system_asym .= imag(lambda)

    if with_control
        sym_operators = [ones(1,1)]
        asym_operators = [ones(1,1)]
    else
        sym_operators = Matrix{Float64}[]
        asym_operators = Matrix{Float64}[]
    end

    return SchrodingerProb(
        system_sym, system_asym,
        sym_operators, asym_operators,
        u0, v0, 
        tf, nsteps,
        N_ess_levels, N_guard_levels
    )

end
