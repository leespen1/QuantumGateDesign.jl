using Plots
using LinearAlgebra: tr, randn, svd

function mean(x)
    sum(x) / length(x)
end

function evolve(U0, H, T)
    return exp(-im*T*H)*U0
end

function infidelity(Uf, V_tg)
    return 1 - (1/size(V_tg,2))*abs(tr(Uf'*V_tg))^2
end

function infidelity_stochastic(Uf, V_tg_vec)
    individual_infidelities = [infidelity(Uf, V_tg) for V_tg in V_tg_vec]
    return mean(individual_infidelities)
end

"""
Make a random skew hermitan matrix.
"""
function random_skew_hermitian(n, gaussian=false)
    if gaussian
        A = randn(ComplexF64, n, n) # Make random matrix, with entries mean=0, stddev=1
    else # Alternatively, use a uniform distribution
        A_real = rand(n, n)  .- 0.5
        A_imag = rand(n, n)  .- 0.5
        A = A_real + im*A_imag
    end
    return (A - A') # Make A skew-hermitian
end

"""
Project A onto the nearest unitary (distance measured Frobenius norm)
Useful in case I really want to ensure the 
"""
function project_onto_unitary(A)
    A_svd = svd(A)
    return A_svd.U * A_svd.V
end

"""
Function to create a unitary matrix near the original unitary matrix U. The
distance between the original and nearby unitaries seems to be ~epsilon.
"""
function nearby_unitary(U, epsilon=1e-3; gaussian=false)
    n = size(U, 1)
    X = epsilon*random_skew_hermitian(n, gaussian)
    V = exp(X) # V is unitary since X is skew-Hermitian
    return U * V # The product of unitary matrices is unitary
end

a = [0 1;0 1]
H(θ) = θ*(a + a')
U0 = [1 0;0 1]
V_tg = [0 1;0 1]
T = 1

epsilon=1

xs = LinRange(0, 2.5, 100)
ys = [infidelity(evolve(U0, H(x), T), V_tg) for x in xs]
pl = plot(xs, ys, label="Normal")
plot!(pl, ylabel="Infidelity", xlabel="Control Amplitude")
plot!(pl, title="epsilon=$epsilon")

N_targets = 1000
V_tg_vec = [nearby_unitary(V_tg, epsilon, gaussian=false) for i in 1:N_targets]
ys_sgd = [infidelity_stochastic(evolve(U0, H(x), T), V_tg_vec) for x in xs]
plot!(pl, xs, ys_sgd, label="Stochastic")

