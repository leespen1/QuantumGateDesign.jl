using QuantumGateDesign, Random, IterativeSolvers, LinearAlgebra, Enzyme, Zygote

function coeff(j,p,q) 
    return factorial(p)*factorial(p+q-j)/(factorial(p+q)*factorial(p-j))
end

"""
"""
function hard_coded_eval_forward(prob::SchrodingerProb, pcof::AbstractVector{<: Real}, order::Integer)
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

    return UT
end

function calc_infidelity(prob::SchrodingerProb, pcof::AbstractVector{<: Real}, order::Integer, target)
    UT = hard_coded_eval_forward(prob, pcof, order)
    return QuantumGateDesign.infidelity_real(UT, target, prob.N_initial_conditions)
end

rand_prob_size =11
rand_tf = 10.0
rand_prob_N_operators = 1
rand_nsteps = 1
order = 2

rand_prob = QuantumGateDesign.construct_rand_prob(
    rand_prob_size, rand_prob_N_operators, tf=rand_tf, nsteps=rand_nsteps,
    gmres_abstol=1e-16, gmres_reltol=0
)
target = rand(rand_prob.real_system_size, rand_prob.N_initial_conditions)
controls = [GRAPEControl(1, rand_tf) for _ in 1:rand_prob_N_operators]

pcof = rand(MersenneTwister(0), 2*rand_prob_N_operators)

UT_hard = hard_coded_eval_forward(rand_prob, pcof, order)
println("Finished hard-coded")
history = eval_forward(rand_prob, controls, pcof, order=order)
println("Finished soft-coded")
UT_soft = vcat(real(history[:,end,:]), imag(history[:,end,:]))
@show norm(UT_hard - UT_soft)
#@assert isapprox(UT_hard, UT_soft, rtol=1e-15)

@show calc_infidelity(rand_prob, pcof, order, target)
f(x) = calc_infidelity(rand_prob, x, order, target)

dpcof = zeros(size(pcof))
#Enzyme.autodiff(Reverse, calc_infidelity, Active, Const(rand_prob), Duplicated(pcof, dpcof), Const(order), Const(target));
#Enzyme.autodiff(set_runtime_activity(Reverse), calc_infidelity, Active, Const(rand_prob), Duplicated(pcof, dpcof), Const(order), Const(target));
zygote_grad = Zygote.gradient(f, pcof)
discrete_adjoint_grad = discrete_adjoint(rand_prob, controls, pcof, real_to_complex(target), order=order)
forced_grad = eval_grad_forced(rand_prob, controls, pcof, real_to_complex(target), order=order)

@show zygote_grad
@show discrete_adjoint_grad
@show forced_grad
