# For Filon
include("/home/spencer/High_Frequency_Research/filon_research_github/Tools/scalar_matrix.jl")
include("/home/spencer/High_Frequency_Research/filon_research_github/Src/master.jl")

# For Hermite
include("./high_order_2d.jl")

using LinearAlgebra
using LaTeXStrings


function main()
    t0 = 0.0
    tf = 20.0
    tspan = (t0, tf)
    n_steps_ODEProblem = 10
    dt_save = (tspan[2] - tspan[1])/n_steps_ODEProblem

    Q0 = [1.0 0.0;
          0.0 1.0;
          0.0 0.0;
          0.0 0.0]
    #Q0 = zeros(4,1)
    #Q0[2,1] = 1
    num_RHS = size(Q0, 2)

    p = 1e-2 # (want control to be small for filon to work well)

    true_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)
    for i in 1:num_RHS
        Q0_col = Q0[:,i]
        ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0_col, tspan, p) # Need this option for debug to work
        data_solution = DifferentialEquations.solve(ODE_prob, saveat=dt_save, abstol=1e-15, reltol=1e-15)
        data_solution_mat = Array(data_solution)
        true_sol_ary[:,i,:] .= data_solution_mat
    end

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Stt(t,a) = [0.0 0.0;
                 0.0 0.0]
    Ktt(t,a) = [0.0 -a*cos(t);
               -a*cos(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]


    N = 5
    nsteps = n_steps_ODEProblem .* (2 .^ (0:N)) # Double the number of steps each time
    max_ratios = zeros(N)
    min_ratios = zeros(N)
    max_errors = zeros(N+1)
    ft_errors = zeros(N+1)
    all_t_errors = zeros(N+1)

    # 2nd order schrodinger
    schroprob = SchrodingerProb(tspan, nsteps[1], Q0, p, S, K, St, Kt, Stt, Ktt, Sa, Ka)
    Qs_prev = eval_forward(schroprob, p; order=2)[:,:,1:nsteps[1]÷n_steps_ODEProblem:end]
    Q_diffs_prev = (Qs_prev - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((Qs_prev - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((Qs_prev - true_sol_ary))

    for i in 2:N+1
        schroprob.n_timesteps = nsteps[i]
        Qs_next = eval_forward(schroprob, p; order=2)[:,:,1:nsteps[i]÷n_steps_ODEProblem:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        ft_errors[i] = norm((Qs_next - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((Qs_next - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end

    pl1 = plot(nsteps[2:end], min_ratios, label="Hermite-2")
    plot!(pl1, xlabel="# Timesteps", ylabel="Log2 E(2*Δt)/E(Δt)")
    plot!(pl1, xscale=:log10)
    plot!(pl1, title="Convergence")
    plot!(pl1, yticks=[0,2,4,6,8])

    pl2 = plot(nsteps, max_errors, label="Hermite-2")
    plot!(pl2, xlabel="# Timesteps", ylabel="Max E(Δt) (per entry)")
    plot!(pl2, scale=:log10)
    plot!(pl2, title="Error")

    pl3 = plot(nsteps, ft_errors, label="Hermite-2")
    plot!(scale=:log10)
    plot!(xlabel="# Timesteps", ylabel="Error at t=T")

    linewidth = 4
    markersize = 10
    pl4 = plot(tf ./ nsteps, all_t_errors, color=:Blue, markershape=:circle, label="",
               dpi=100, size=(14*100,9*100),
               legendfontsize=33,guidefontsize=33,tickfontsize=33,
               linewidth=linewidth, markersize=markersize, margin=0.5*Plots.inch)
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:Blue, markershape=:circle, markersize=markersize, label="Hermite-2")

    plot!(scale=:log10)
    plot!(xlabel=L"\Delta t", ylabel=L"|| U_{\Delta t} - U_{True} ||_2")
    #plot!(xlabel=L"\Delta t")
    plot!(yticks= [1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    plot!(xticks= [1.0, 10.0^(-0.5), 10.0^(-1),10.0^(-1.5), 10.0^(-2), 10.0^(-2.5),10.0^(-3.0)])
    plot!(legend=:bottomright)



    # 4th order schrodinger
    schroprob.n_timesteps = nsteps[1]
    Qs_prev = eval_forward(schroprob, p; order=4)[:,:,1:nsteps[1]÷n_steps_ODEProblem:end]
    Q_diffs_prev = (Qs_prev - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((Qs_prev - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((Qs_prev - true_sol_ary))

    for i in 2:N+1
        schroprob.n_timesteps = nsteps[i]
        Qs_next = eval_forward(schroprob, p; order=4)[:,:,1:nsteps[i]÷n_steps_ODEProblem:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        ft_errors[i] = norm((Qs_next - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((Qs_next - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end

    plot!(pl1, nsteps[2:end], min_ratios, label="Hermite-4")
    plot!(pl2, nsteps, max_errors, label="Hermite-4")
    plot!(pl3, nsteps, ft_errors, label="Hermite-4")
    plot!(pl4, tf ./ nsteps, all_t_errors, color=:Blue, markershape=:utriangle, linewidth=linewidth, markersize=markersize, label="")
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:Blue, markershape=:utriangle, linewidth=linewidth, markersize=markersize, label="Hermite-4")


    # 6th order schrodinger
    schroprob.n_timesteps = nsteps[1]
    Qs_prev = eval_forward(schroprob, p; order=6)[:,:,1:nsteps[1]÷n_steps_ODEProblem:end]
    Q_diffs_prev = (Qs_prev - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((Qs_prev - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((Qs_prev - true_sol_ary))

    for i in 2:N+1
        schroprob.n_timesteps = nsteps[i]
        Qs_next = eval_forward(schroprob, p; order=6)[:,:,1:nsteps[i]÷n_steps_ODEProblem:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end]
        Q_diffs_next = (Qs_next - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        ft_errors[i] = norm((Qs_next - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((Qs_next - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end

    plot!(pl1, nsteps[2:end], min_ratios, label="Hermite-6")
    plot!(pl2, nsteps, max_errors, label="Hermite-6")
    plot!(pl3, nsteps, ft_errors, label="Hermite-6")
    plot!(pl4, tf ./ nsteps, all_t_errors, color=:Blue, markershape=:square, linewidth=linewidth, markersize=markersize,  label="")
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:Blue, markershape=:square, linewidth=linewidth, markersize=markersize,  label="Hermite-6")


    #Filon 2
    #ω = [0.0, 1.0]
    ω = [0.0, 1.0]
    # u0 defined above

    H(t) = im .* [0.0 p*cos(t); p*cos(t) 1.0]
    #H(t) = im * [0.0 0.0; 0.0 1.0]
    f(t, u) = H(t)*u
    filon_prob_col1 = ImplicitFilonProblem2(Q0[1:2,1] + im*Q0[3:4,1], t0, ω, H)
    filon_prob_col2 = ImplicitFilonProblem2(Q0[1:2,2] + im*Q0[3:4,2], t0, ω, H)

    tgrid_filon = LinRange(t0, tf, n_steps_ODEProblem+1)
    filon_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)

    sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)
    sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)

    filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
    filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
    filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
    filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

    filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
    filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
    filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
    filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

    Q_diffs_prev = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((filon_sol_ary - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((filon_sol_ary - true_sol_ary))

    for i in 2:N+1
        sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)
        sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)

        filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
        filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
        filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
        filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

        filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
        filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
        filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
        filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

        Q_diffs_next = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        #ft_errors[i] = norm((filon_sol_ary - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((filon_sol_ary - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end
    plot!(pl1, nsteps[2:end], min_ratios, label="Filon-2")
    plot!(pl2, nsteps, max_errors, label="Filon-2")
    plot!(pl3, nsteps, ft_errors, label="Filon-2")
    plot!(pl4, tf ./ nsteps, all_t_errors, color=:Red, markershape=:circle, linewidth=linewidth, markersize=markersize,  label="")
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:Red, markershape=:circle, linewidth=linewidth, markersize=markersize,  label="Filon-2")


    # Filon 4
    filon_prob_col1 = ImplicitFilonProblem(Q0[1:2,1] + im*Q0[3:4,1], t0, ω, H)
    filon_prob_col2 = ImplicitFilonProblem(Q0[1:2,2] + im*Q0[3:4,2], t0, ω, H)

    tgrid_filon = LinRange(t0, tf, n_steps_ODEProblem+1)
    filon_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)

    sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)
    sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)

    filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
    filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
    filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
    filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

    filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
    filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
    filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
    filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

    Q_diffs_prev = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((filon_sol_ary - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((filon_sol_ary - true_sol_ary))

    for i in 2:N+1
        sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)
        sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)

        filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
        filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
        filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
        filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

        filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
        filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
        filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
        filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

        Q_diffs_next = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        ft_errors[i] = norm((filon_sol_ary - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((filon_sol_ary - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end
    plot!(pl1, nsteps[2:end], min_ratios, label="Filon-4")
    plot!(pl2, nsteps, max_errors, label="Filon-4")
    plot!(pl3, nsteps, ft_errors, label="Filon-4")
    plot!(pl4, tf ./ nsteps, all_t_errors, color=:red, markershape=:utriangle, linewidth=linewidth, markersize=markersize,  label="")
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:red, markershape=:utriangle, linewidth=linewidth, markersize=markersize,  label="Filon-4")


    # Filon 6
    filon_prob_col1 = ImplicitFilonProblem6(Q0[1:2,1] + im*Q0[3:4,1], t0, ω, H)
    filon_prob_col2 = ImplicitFilonProblem6(Q0[1:2,2] + im*Q0[3:4,2], t0, ω, H)

    tgrid_filon = LinRange(t0, tf, n_steps_ODEProblem+1)
    filon_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)

    sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)
    sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[1]÷n_steps_ODEProblem)

    filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
    filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
    filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
    filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

    filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
    filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
    filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
    filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

    Q_diffs_prev = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
    max_errors[1] = max(Q_diffs_prev...)
    ft_errors[1] = norm((Qs_prev - true_sol_ary)[:,:,end])
    all_t_errors[1] = norm((filon_sol_ary- true_sol_ary))

    for i in 2:N+1
        sol_filon_col1 = solve(filon_prob_col1, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)
        sol_filon_col2 = solve(filon_prob_col2, tgrid_filon, nsteps[i]÷n_steps_ODEProblem)

        filon_sol_ary[1,1,:] = real.(sol_filon_col1[1,:])
        filon_sol_ary[3,1,:] = imag.(sol_filon_col1[1,:])
        filon_sol_ary[2,1,:] = real.(sol_filon_col1[2,:])
        filon_sol_ary[4,1,:] = imag.(sol_filon_col1[2,:])

        filon_sol_ary[1,2,:] = real.(sol_filon_col2[1,:])
        filon_sol_ary[3,2,:] = imag.(sol_filon_col2[1,:])
        filon_sol_ary[2,2,:] = real.(sol_filon_col2[2,:])
        filon_sol_ary[4,2,:] = imag.(sol_filon_col2[2,:])

        Q_diffs_next = (filon_sol_ary - true_sol_ary)[:,:,2:end] # Remove first entry, initial condition
        ratios = log2.(abs.(Q_diffs_prev ./ Q_diffs_next))
        max_ratios[i-1] = max(ratios...)
        min_ratios[i-1] = min(ratios...)
        max_errors[i] = max(Q_diffs_next...)
        #ft_errors[i] = norm((filon_sol_ary - true_sol_ary)[:,:,end])
        all_t_errors[i] = norm((filon_sol_ary - true_sol_ary))
        
        Q_diffs_prev .= Q_diffs_next
    end
    plot!(pl1, nsteps[2:end], min_ratios, label="Filon-6")
    plot!(pl2, nsteps, max_errors, label="Filon-6")
    plot!(pl3, nsteps, ft_errors, label="Filon-6")
    plot!(pl4, tf ./ nsteps, all_t_errors, color=:red, markershape=:square, linewidth=linewidth, markersize=markersize,  label="")
    scatter!(pl4, tf ./ nsteps, all_t_errors, color=:red, markershape=:square, linewidth=linewidth, markersize=markersize,  label="Filon-6")

    plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 2, color=:gray, linewidth=linewidth-1, label=L"\Delta t^2")
    plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 4, color=:gray, linestyle=:dash, linewidth=linewidth-1, label=L"\Delta t^4")
    plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 6, color=:gray, linestyle=:dashdot, linewidth=linewidth-1, label=L"\Delta t^6")
    #plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 2, color=:gray, linewidth=linewidth-1, label="")
    #plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 4, color=:gray, linestyle=:dash, linewidth=linewidth-1, label="")
    #plot!(pl4, tf ./ nsteps, (tf ./ nsteps) .^ 6, color=:gray, linestyle=:dashdot, linewidth=linewidth-1, label="")
    plot!(pl4, legend=:bottomright)

    scatter!(pl4, color=:Red, markershape=:circle, markersize=markersize, label="Hermite-8")

    pl = plot(pl1, pl2, layout=(1,2))
    plot!(pl, plot_title="Error and Convergence (Per Entry)")
    #return pl
    return pl1, pl2, pl3, pl4
end

function main2()
    t0 = 0.0
    tf = 20.0
    tspan = (t0, tf)
    n_steps_ODEProblem = 1000
    dt_save = (tspan[2] - tspan[1])/n_steps_ODEProblem
    tgrid_true = LinRange(t0,tf,n_steps_ODEProblem+1)

    #Q0 = [0.0;
    #      1.0;
    #      0.0;
    #      0.0]
    Q0 = zeros(4,1)
    Q0[2,1] = 1
    num_RHS = size(Q0, 2)

    p = 1e-2 # (want control to be small for filon to work well)

    true_sol_ary = zeros(4,num_RHS,n_steps_ODEProblem+1)
    for i in 1:num_RHS
        Q0_col = Q0[:,i]
        ODE_prob = ODEProblem{true, SciMLBase.FullSpecialize}(schrodinger!, Q0_col, tspan, p) # Need this option for debug to work
        data_solution = DifferentialEquations.solve(ODE_prob, saveat=dt_save, abstol=1e-14, reltol=1e-14)
        data_solution_mat = Array(data_solution)
        true_sol_ary[:,i,:] .= data_solution_mat
    end

    S(t,a) = [0.0 0.0;
               0.0 0.0]
    K(t,a) = [0.0 a*cos(t);
               a*cos(t) 1.0]
    St(t,a) = [0.0 0.0;
                 0.0 0.0]
    Kt(t,a) = [0.0 -a*sin(t);
               -a*sin(t) 0.0]
    Stt(t,a) = [0.0 0.0;
                 0.0 0.0]
    Ktt(t,a) = [0.0 -a*cos(t);
               -a*cos(t) 0.0]
    Sa(t,a) = [0.0 0.0;
               0.0 0.0]
    Ka(t,a) = [0.0 cos(t);
               cos(t) 0.0]

    n_filon_steps = 3
    n_hermite_steps = 6
    n_rk4_steps = 12
    # Hermite
    tgrid_hermite = LinRange(t0, tf, n_hermite_steps+1)
    schroprob = SchrodingerProb(tspan, n_hermite_steps, Q0, p, S, K, St, Kt, Stt, Ktt, Sa, Ka)
    Qs = eval_forward(schroprob, p; order=4)

    # Filon 2
    #ω = [0.0, 1.0]
    ω = [0.0, 1.0]
    # u0 defined above

    H(t) = im .* [0.0 p*cos(t); p*cos(t) 1.0]
    #H(t) = im * [0.0 0.0; 0.0 1.0]
    f(t, u) = H(t)*u
    filon_prob = ImplicitFilonProblem(Q0[1:2,1] + im*Q0[3:4,1], t0, ω, H)
    tgrid_filon = LinRange(t0, tf, n_filon_steps+1)
    sol_filon = solve(filon_prob, tgrid_filon)

    rk4_prob = RK4Problem(Q0[1:2,1] + im*Q0[3:4,1], t0, f)
    tgrid_rk4 = LinRange(t0, tf, n_rk4_steps+1)
    sol_rk4 = solve(rk4_prob, tgrid_rk4)


    markersize = 10
    p3 = plot(tgrid_true, true_sol_ary[2,1,:], label="True Solution", dpi=100, size=(14*100,9*100),
             legendfontsize=33,guidefontsize=33,tickfontsize=33,
             linewidth=4, margin=0.5*Plots.inch)
    scatter!(tgrid_hermite, Qs[2,1,:], color=:Blue, markershape=:circle, markersize=markersize, label="Hermite-4")
    scatter!(tgrid_filon, real(sol_filon[2,:]), color=:Red, markershape=:utriangle,markersize=markersize, label="Filon-4")
    scatter!(tgrid_rk4, real(sol_rk4[2,:]), color=:Magenta, markershape=:square,markersize=markersize, label="Explicit RK-4")
    plot!(xlabel="t")


    return p3
end

