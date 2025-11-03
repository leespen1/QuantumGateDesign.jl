using QuantumGateDesign, Random, LinearAlgebra, DelimitedFiles

function main()
    degree = 14
    seed = 0
    atol = 1e-15
    rtol = 1e-15
    D1 = 16
    N_osc_levels = 10
    Tmax = 550.0
    start_in_highest_state = false

    cnot3ret = QuantumGateDesign.setup_cnot3(
        seed=seed,
        atol=atol,
        rtol=rtol,
        D1=D1,
        N_osc_levels=N_osc_levels,
        Tmax=Tmax
    )


    if start_in_highest_state # Change initial conditions to start in highest state
        cnot3ret.qgd_prob.u0 = zeros(cnot3ret.qgd_prob.N_tot_levels, 1)
        cnot3ret.qgd_prob.v0 = zeros(cnot3ret.qgd_prob.N_tot_levels, 1)
        cnot3ret.qgd_prob.u0[end,1] = 1
        cnot3ret.qgd_prob.N_ess_levels = 1
        cnot3ret.qgd_prob.N_initial_conditions = 1

        cnot3ret.target = cnot3ret.qgd_prob.u0 + im .* cnot3ret.qgd_prob.v0 # Use initial state as final state
        cnot3ret.target[end,1] = 1

    end


    controls = get_controls(degree, D1, cnot3ret.juqbox_params.Cfreq, cnot3ret.tf)

    N_coeff = QuantumGateDesign.get_number_of_control_parameters(controls)

    # Coefficients uniformly distributed between amax and -amax
    #pcof = cnot3ret.amax * 2 * (0.5 .- rand(MersenneTwister(seed), N_coeff))
    pcof_history = readdlm("targetError=1e-7_cnot3OptimizationTest_order=6_degree=14_seed=5_nsteps=5409_atol=1.0e-15_rtol=1.0e-15_D1=16_time=6.0_maxiter=10000_nthreads=4_costType=GeneralizedInfidelity_gateDuration=550.0_nCavityLevels=10_pcof.csv", ',')
    pcof = pcof_history[end,:]


    orders_vec = [2, 4, 6, 8, 10, 12]
    #orders_vec = [2]
    nsteps_vec = 2 .^ (5:15)
    unitary_deviations_mat = Matrix{Float64}(undef, length(nsteps_vec), length(orders_vec))


    prob = cnot3ret.qgd_prob
    filename = "unitarydeviation_vs_order_nsteps.dlm"
    for (i, nsteps) in enumerate(nsteps_vec)
        for (j, order) in enumerate(orders_vec)
            println("order=$order, nsteps=2^$(log2(nsteps))")
            prob.nsteps = nsteps
            t_grid = LinRange(0, prob.tf, 1+nsteps)
            history = eval_forward(prob, controls, pcof, order=order)
            unitary_deviations_vec = mapslices(U -> norm(U'*U - LinearAlgebra.I), history, dims=(1,3)) |> vec
            unitary_deviation = sqrt(sum(x -> x^2, unitary_deviations_vec) / length(unitary_deviations_vec))
            unitary_deviations_mat[i,j] = unitary_deviation
        end
        # Write matrix to dlm
        open(filename, "w") do io
            writedlm(io, unitary_deviations_mat[1:i, :])
        end
    end

    #=
    plot = false
    if plot
        fig = Figure()
        ax = Axis(
            fig[1, 1];
            title="Unitary Deviation",
            xlabel="t (nanoseconds)",
            ylabel="||UdaggerU - I||_F",
            xscale=CairoMakie.log2,
            yscale=CairoMakie.log10,
            limits=((2^5, 2^20), (1e-10, 1e0)),
        )
        for (i, order) in enumerate(orders_vec)
            lines!(ax, nsteps_vec, unitary_deviations_mat[i, :], label="Order $order")
        end

        Legend(fig[end+1,:], ax, orientation = :horizontal, tellwidth = false, nbanks=2, framevisible=false)
        save("unitarydeviation_plot.png", fig)
        save("unitarydeviation_plot.svg", fig)
        save("unitarydeviation_plot.pdf", fig)
    end
    =#

    return unitary_deviations_mat
end

main()

