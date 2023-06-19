using Test

prob = gargamel_prob()
pcof = rand(4)

history = eval_forward(prob, pcof)

target = rand(4,2)
eval_grad_finite_difference(prob, target, alpha, cost_type=:Infidelity)
eval_grad_finite_difference(prob, target, alpha, cost_type=:Tracking)
eval_grad_finite_difference(prob, target, alpha, cost_type=:Norm)
