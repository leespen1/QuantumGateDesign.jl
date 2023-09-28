# This object may not be necessary. There really isn't much to put here.
struct OptimizationParameters
    N_essential_levels::Int64
    N_guard_levels::Int64
end

function initial_state(N_ess::Int, N_guard::Int)
    u0 = zeros(N_ess+N_guard, N_ess)
    for i=1:N_ess
        u0[i,i] = 1
    end
    v0 = zeros(N_ess+N_guard, N_ess)
    return u0, v0
end

function target_helper(V_complex::AbstractMatrix, N_ess::Int, N_guard::Int)
    @assert size(V_complex, 1) == size(V_complex, 2) == N_ess
    N_tot = N_ess + N_guard
    V_real = zeros(2*N_tot, N_ess)

    V_real[1:N_ess,:] .= real(V_complex)
    # Should this be negative? What did we decide on?
    V_real[N_tot+1:N_tot+N_ess, :] .= imag(V_complex)

    return  V_real
end
