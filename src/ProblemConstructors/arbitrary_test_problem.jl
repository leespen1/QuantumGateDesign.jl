function arbitrary_test_problem(N_ess_levels::Int, N_guard_levels::Int;
        tf::Float64=1.0, nsteps::Int64=10
    )

    N_tot_levels = N_ess_levels + N_guard_levels

    system_sym = zeros(N_tot_levels, N_tot_levels)
    system_asym = zeros(N_tot_levels, N_tot_levels)

    el = 1
    for row_i in 1:N_tot_levels
        for col_i in row_i:N_tot_levels
            system_sym[row_i, col_i] = el
            system_sym[col_i, row_i] = el
            if (row_i != col_i)
                system_asym[row_i, col_i] = el
                system_asym[col_i, row_i] = -el
            end
            el += 1
        end
    end

end
