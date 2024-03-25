using QuantumGateDesign
using Plots
using Random
# Single qubit in the rotating frame
system_sym = zeros(2,2)
system_sym[1,1] = 0.84
system_sym[2,2] = 1.78
system_asym = zeros(2,2)

a = [0.0 1; 0 0]
sym_ops = [a + a']
asym_ops = [a - a']

u0 = [1.0 0; 0 1]
v0 = [0.0 0; 0 0]

tf = 3.5 # Change this
nsteps = 500
N_ess_levels = 2

prob = SchrodingerProb(
    system_sym, 
    system_asym,
    sym_ops,
    asym_ops,
    u0,
    v0,
    tf,
    nsteps,
    N_ess_levels
)

# Swap gate, analytic solution should be a pi/2-pulse
# If p = Re(Ω)cos(θ) + Im(Ω)sin(θ), then tf = pi/(2*|Ω|)
target = [0.0 1.0
          1   0
          0   0
          0   0]

plot_heat = false
if plot_heat
    control = QuantumGateDesign.GRAPEControl(1, prob.tf+0.1) # For now, it doesn't matter since amplitudes last for whole duration

    N_samples = 31
    var_range = LinRange(-2, 2, N_samples)
    var1s = zeros(N_samples, N_samples)
    var2s = zeros(N_samples, N_samples)
    infidelities = zeros(N_samples, N_samples)

    for (j, var1) in enumerate(var_range)
        for (k, var2) in enumerate(var_range)
            var1s[j,k] = var1
            var2s[j,k] = var2
            pcof = [var1, var2]
            infidelities[j,k] = infidelity(prob, control, pcof, target)
        end
    end


    pl = heatmap(var_range, var_range, infidelities, color=:thermal,
                xlabel="θ₁", ylabel="θ₂", colorbar_title="Infidelity")
end

control = QuantumGateDesign.bspline_control(prob.tf, 4, [0.84, 1.78])

function stochastic_sgd(prob, control, pcof_orig, target, learning_rate, N_epochs=10)
    N_coeff = control.N_coeff
    pcof_grad_desc = copy(pcof_orig)
    pcof_sgd = copy(pcof_orig)

    grad_desc_infidelities = []
    sgd_infidelities = []
    gd_pcofs = []
    sgd_pcofs = []

    for n in 1:N_epochs
        grad = discrete_adjoint(prob, control, pcof_grad_desc, target)
        pcof_grad_desc = pcof_grad_desc - learning_rate*grad
        this_infidelity = infidelity(prob, control, pcof_grad_desc, target)
        push!(grad_desc_infidelities, this_infidelity)
        push!(gd_pcofs, copy(pcof_grad_desc))

    end

    for n in 1:N_epochs
        for init_i in randperm(prob.N_initial_conditions)
            vec_prob = QuantumGateDesign.VectorSchrodingerProb2(prob, init_i)
            vec_targ = reshape(target[:, init_i], :, 1)
            grad = discrete_adjoint(vec_prob, control, pcof_sgd, vec_targ)
            pcof_sgd = pcof_sgd - learning_rate*grad
            # Infidelity still computed w.r.t whole basis
        end
        this_infidelity = infidelity(prob, control, pcof_sgd, target)
        push!(sgd_infidelities, this_infidelity)
        push!(sgd_pcofs, copy(pcof_sgd))
    end

    return grad_desc_infidelities, sgd_infidelities, gd_pcofs, sgd_pcofs
end

#learning_rate = 0.2
learning_rate = 0.15
N_epochs = 20
Random.seed!(152)
my_gd_infs = zeros(N_epochs)
my_sgd_infs = zeros(N_epochs)
good_pcof_orig = zeros(N_coeff)
for dummy in 1:25
    pcof_orig = rand(N_coeff)
    grad_desc_infidelities, sgd_infidelities, ~, ~ = stochastic_sgd(
        prob, control, pcof_orig, target, learning_rate, N_epochs
    )
    my_gd_infs .= grad_desc_infidelities
    my_sgd_infs .= sgd_infidelities


    if sgd_infidelities[end] < grad_desc_infidelities[end]
        println("Run $dummy")
        global good_pcof_orig .= pcof_orig
        #println("Pcof orig: \n", pcof_orig)
        println("sgd: $(sgd_infidelities[end]), gd: $(grad_desc_infidelities[end])")
        break
    end
end

ret = stochastic_sgd(prob, control, good_pcof_orig, target, learning_rate, N_epochs)
pl2 = plot(ret[1], label="Gradient Descent", lw=2, markershape=:cross)
plot!(pl2, ret[2], label="Stochastic Gradient Descent", lw=2, markershape=:cross)

plot!(pl2, xlabel="# Epochs", ylabel="Infidelity")


param1range = LinRange(0.5, 1, 6)
param2range = LinRange(1.5, 2, 6)
infidelities_2 = zeros(6,6)
pcof_start = rand(N_coeff)
for (i, param1) in enumerate(param1range)
    for (j, param2) in enumerate(param2range)
        local_control = QuantumGateDesign.bspline_control(prob.tf, 4, [param1, param2])


        N_coeff = control.N_coeff
        pcof_grad_desc = copy(pcof_start)

        grad_desc_infidelities = []
        gd_pcofs = []

        for n in 1:10
            grad = discrete_adjoint(prob, local_control, pcof_grad_desc, target)
            pcof_grad_desc = pcof_grad_desc - learning_rate*grad
            this_infidelity = infidelity(prob, local_control, pcof_grad_desc, target)
            push!(grad_desc_infidelities, this_infidelity)
            push!(gd_pcofs, copy(pcof_grad_desc))
        end

        infidelities_2[i,j] = grad_desc_infidelities[end]
    end
end

pl3 = heatmap(param1range, param2range, infidelities_2)
