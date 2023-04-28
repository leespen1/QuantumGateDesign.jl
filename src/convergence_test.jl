function convergence_test!(prob::SchrodingerProb, α=missing; base_nsteps=2, N=5, cost_type=:Infidelity)
    orders = [2,4]
    sol_errs = zeros(N, length(orders))
    costs = zeros(N, length(orders))

    step_sizes = zeros(N)
    for n in 1:N
        step_sizes[n] = prob.tf / (base_nsteps^n)
    end

    # Get 'true' solution, using most timesteps and highest order
    prob.nsteps = base_nsteps^(N+1)
    history = eval_forward(prob, α, order=max(orders...)) 
    final_state_fine = history[:,end]

    for j in 1:length(orders)
        order = orders[j]
        final_states = zeros(4,N)
        for i in 1:N
            prob.nsteps = base_nsteps^i
            history = eval_forward(prob, α, order=order)
            final_states[:,i] = history[:,end]
        end

        for i in 1:N
            sol_errs[i,j] = norm(final_states[:,i] - final_state_fine)
            if cost_type == :Infidelity
                costs[i,j] = infidelity(final_states[:,i], final_state_fine)
            elseif cost_type == :Norm
                costs[i,j] = 0.5*norm(final_states[:,i])^2
            else
                throw("Invalid cost type: $cost_type")
            end

        end
    end
   
    return step_sizes, sol_errs, costs, orders
end


function plot_convergence_test(step_sizes, sol_errs, infidelities, orders)
    pl = plot()
    #colors = [:blue, :red]
    for i in 1:length(orders)
        plot!(pl, step_sizes, abs.(sol_errs[:,i]), linewidth=2, marker=:circle, label="Error (Order $(orders[i]))", color=i)
        plot!(pl, step_sizes, abs.(infidelities[:,i]), linewidth=2, marker=:square, label="Infidelities (Order $(orders[i]))", color=i)
    end
    plot!(pl, step_sizes, step_sizes .^ 2, label="Δt^2", linestyle=:solid, color=:grey, lw=2)
    plot!(pl, step_sizes, step_sizes .^ 4, label="Δt^4", linestyle=:dash, color=:grey, lw=2)
    plot!(pl, step_sizes, step_sizes .^ 6, label="Δt^6", linestyle=:dashdot, color=:grey, lw=2)
    plot!(pl, legendfontsize=14, guidefontsize=14, tickfontsize=14)
    plot!(pl, scale=:log10)
    plot!(pl, legend=:bottomright)
    plot!(pl, xlabel="Δt")
    return pl
end
