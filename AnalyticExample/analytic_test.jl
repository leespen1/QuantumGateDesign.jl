using QuantumGateDesign, Random, IterativeSolvers, LinearAlgebra, Zygote

function coeff(j::Integer,p::Integer,q::Integer) 
    return factorial(p)*factorial(p+q-j)/(factorial(p+q)*factorial(p-j))
end

function identity_mat(n::Integer)
    id_mat = zeros(n,n)
    for i = 1:n
        id_mat[i,i] = 1
    end
    return id_mat
end

"""
"""
function hard_coded_eval_forward(prob::SchrodingerProb, pcof::AbstractVector{<: Real}, order::Integer, target=missing)
    q = div(order, 2)
    #@show q
    #@assert length(pcof) == 2*prob.N_operators

    Kd = prob.system_sym
    Sd = prob.system_asym
    Ad = [Sd Kd; -Kd Sd]
    pcof_ps = @view pcof[1:2:end]
    pcof_qs = @view pcof[2:2:end]
    #@show pcof_ps
    #@show pcof_qs
    Ac_ops = [[q*Sc p*Kc; -p*Kc q*Sc] for (Kc, Sc, p, q) in 
              zip(prob.sym_operators, prob.asym_operators, pcof_ps, pcof_qs)]
    #@show Ac_ops

    D = copy(Ad)
    for Ac_op in Ac_ops
        D = D + Ac_op  
    end
    Δt = prob.tf

    LHS = zeros(prob.real_system_size, prob.real_system_size)
    RHS = zeros(prob.real_system_size, prob.real_system_size)

    for j=0:q
        LHS = LHS + ((-1)^j * coeff(j,q,q) * (Δt^j) * (D^j) / factorial(j))
        RHS = RHS + (coeff(j,q,q) * (Δt^j) * (D^j) / factorial(j))
    end

    #@show cond(LHS)
    #@show norm(LHS - QuantumGateDesign.form_LHS_no_control(prob, order))

    U0 = vcat(rand_prob.u0, rand_prob.v0)
    #UT = zeros(rand_prob.real_system_size, rand_prob.N_initial_conditions)
    UT = zeros(rand_prob.real_system_size, 0)

    for i in 1:size(U0, 2)
        #UT_i = gmres(LHS, RHS*U0[:,i], abstol=rand_prob.gmres_abstol, reltol=prob.gmres_reltol)
        UT_i = LHS \ (RHS*U0[:,i])
        UT = hcat(UT, UT_i)
        #UT[:,i] = gmres(LHS, RHS*U0[:,i], abstol=rand_prob.gmres_abstol, reltol=prob.gmres_reltol)
        #UT[:,i] = LHS \ (RHS*U0[:,i])
    end

    if !ismissing(target)
        # Terminal Condition
        # ORIGINAL
        A = target[:,:] # Copy target, converting to matrix if vector (will this code work for vectors?)
        B = vcat(target[1+prob.N_tot_levels:end,:], -target[1:prob.N_tot_levels,:])

        ## NEW (MAY BE WRONG, NOT SURE)
        #A = vcat(target[1:prob.N_tot_levels,:], -target[1+prob.N_tot_levels:end,:])
        #B = vcat(target[1+prob.N_tot_levels:end,:], target[1:prob.N_tot_levels,:])
        
        terminal_RHS = (dot(UT, A)*A + dot(UT, B)*B)
        terminal_RHS *= (2.0/(prob.N_ess_levels^2))

        #@show terminal_RHS
        @show cond(transpose(LHS))

        ΛT = zeros(rand_prob.real_system_size, 0)
        for i in 1:size(U0, 2)
            ΛT_i = transpose(LHS) \ terminal_RHS[:,i]
            ΛT = hcat(ΛT, ΛT_i)
        end

        # Gradient Accumulation
        return UT, ΛT
    end

    return UT
end

function calc_infidelity(prob::SchrodingerProb, pcof::AbstractVector{<: Real}, order::Integer, target)
    UT = hard_coded_eval_forward(prob, pcof, order)
    return QuantumGateDesign.infidelity_real(UT, target, prob.N_initial_conditions)
end

rand_prob_size =11
rand_tf = 1000.0
rand_prob_N_operators = 1
rand_nsteps = 1
order = 2

rand_prob = QuantumGateDesign.construct_rand_prob(
    rand_prob_size, rand_prob_N_operators, tf=rand_tf, nsteps=rand_nsteps,
    gmres_abstol=0, gmres_reltol=1e-20
)
target = rand(MersenneTwister(0), rand_prob.real_system_size, rand_prob.N_initial_conditions)
target_complex = QuantumGateDesign.real_to_complex(target)
controls = [GRAPEControl(1, rand_tf) for _ in 1:rand_prob_N_operators]

pcof = rand(MersenneTwister(0), 2*rand_prob_N_operators)

UT_hard, ΛT_hard = hard_coded_eval_forward(rand_prob, pcof, order, target)
println("Finished hard-coded")

history = zeros(rand_prob.real_system_size, 1+div(order,2), 1+rand_nsteps, rand_prob.N_initial_conditions)
QuantumGateDesign.eval_forward!(history, rand_prob, controls, pcof, order=order)
UT_soft = history[:,1,end,:] 
#ΛT_soft = QuantumGateDesign.compute_terminal_condition(rand_prob, controls, pcof, target, UT_soft, order=order)
ΛT_soft = QuantumGateDesign.compute_terminal_condition(rand_prob, controls, pcof, target, UT_soft, order=order)

Λ_hist_soft = QuantumGateDesign.eval_adjoint(rand_prob, controls, pcof, ΛT_soft)
println("Finished soft-coded")




#UT_soft = vcat(real(history[:,end,:]), imag(history[:,end,:]))
@show norm(UT_hard - UT_soft)
@show norm(ΛT_hard - ΛT_soft)
@show norm(UT_hard - UT_soft)/norm(UT_hard)
@show norm(ΛT_hard - ΛT_soft)/norm(ΛT_hard)
#@assert isapprox(UT_hard, UT_soft, rtol=1e-15)

@show calc_infidelity(rand_prob, pcof, order, target)
f(x) = calc_infidelity(rand_prob, x, order, target)
@show QuantumGateDesign.calc_cost_function(rand_prob, controls, pcof, order, target_complex, :Infidelity)
#f(x) = QuantumGateDesign.calc_cost_function(rand_prob, controls, x, order, target, :Infidelity)

#=
dpcof = zeros(size(pcof))
zygote_grad = Zygote.gradient(f, pcof)[1]
discrete_adjoint_grad = discrete_adjoint(rand_prob, controls, pcof, real_to_complex(target), order=order)
forced_grad = eval_grad_forced(rand_prob, controls, pcof, real_to_complex(target), order=order)

@show norm(zygote_grad)
@show norm(discrete_adjoint_grad)
@show norm(forced_grad)
@show norm(zygote_grad - discrete_adjoint_grad)
@show norm(zygote_grad - forced_grad)
@show norm(discrete_adjoint_grad - forced_grad)
@show norm(zygote_grad - discrete_adjoint_grad) / norm(zygote_grad)
@show norm(zygote_grad - forced_grad) / norm(zygote_grad)
@show norm(discrete_adjoint_grad - forced_grad) / norm(forced_grad)
=#
