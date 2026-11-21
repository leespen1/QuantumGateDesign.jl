#=
Cross-validate QuantumGateDesign.jl against QuantumOptics.jl and QuTiP on the
dispersive cQED CNOT system of the JCP paper (Sec. 4.2): two 4-level transmons
coupled through a 10-level resonator bus, in the rotating frame, driven by a
smooth (seeded, random) B-spline carrier-wave control on every subsystem.

Each package builds the Hamiltonian from its own operators (QGD's
`dispersive_qudits_problem`, QuantumOptics.jl's `destroy`/`embed`, QuTiP's
`destroy`/`tensor`), and propagates the four essential basis states over the
full 550 ns gate with its own time integrator:
  - QuantumGateDesign.jl: high-order Hermite method (GMRES + preconditioner)
  - QuantumOptics.jl:     adaptive Runge-Kutta via OrdinaryDiffEq (DP5), tol 1e-14
  - QuTiP:                sesolve with the Verner 9 method, tol 1e-12
(QuTiP's default Adams method agrees only to about 1e-7 at tol 1e-12, so the
higher-order method is used.) A full run takes a few minutes, mostly in DP5.
The control pulse values p_k(t), q_k(t) are evaluated by QGD in every case, so
the comparison checks Hamiltonian construction and time integration, not the
B-spline evaluation.

Run from the repository root:
    julia --project=examples/cross_validation examples/cross_validation/cross_validate_cnot3.jl
The first run installs QuTiP into a private Conda environment (CondaPkg.toml).
=#

using QuantumGateDesign
using LinearAlgebra: Symmetric, norm
using Random: MersenneTwister
using Printf: @printf
import QuantumOptics as QO
using PythonCall: pyimport, pyeval, pyconvert, pylist, pydict

#==============================================================================
System: identical parameters to examples/cnot3_example.jl (from Juqbox.jl's
CNOT3 example). Subsystem order is (resonator, qudit b, qudit a); frequencies
are in GHz and are multiplied by 2pi when building Hamiltonians.
==============================================================================#
subsystem_sizes = (10, 4, 4)
essential_subsystem_sizes = (1, 2, 2)
Tmax = 550.0

fa, fb, fs = 4.10595, 4.81526, 7.8447
xa = 2 * 0.1099
xb = 2 * 0.1126
xs = 0.002494^2 / xa
xab = 1.0e-6
xas = sqrt(xa * xs)
xbs = sqrt(xb * xs)

transition_freqs = (fs, fb, fa)
rotation_freqs = transition_freqs # rotating frame: no bare-frequency terms remain
kerr_coeffs = Symmetric([xs  xbs xas;
                         xbs xb  xab;
                         xas xab xa], :U)

function qgd_problem(nsteps)
    dispersive_qudits_problem(
        subsystem_sizes, essential_subsystem_sizes, transition_freqs,
        rotation_freqs, kerr_coeffs, Tmax, nsteps;
        gmres_abstol=1e-15, gmres_reltol=1e-15,
    )
end

#==============================================================================
Controls: one control per subsystem, a degree-14 B-spline envelope with 16
basis functions on carrier waves at the paper's carrier frequencies (0 and the
cross-Kerr / self-Kerr shifts, as angular frequencies). Coefficients are random
with the same magnitude as the optimized CNOT controls (about 1e-2).
==============================================================================#
degree, N_basis = 14, 16
carrier_freqs = [
    2pi .* [0.0, -xas, -xbs],  # resonator
    2pi .* [0.0, -xb, -xbs],   # qudit b
    2pi .* [0.0, -xa, -xas],   # qudit a
]
controls = [CarrierControl(FortranBSplineControl(degree, N_basis, Tmax), w)
            for w in carrier_freqs]
N_coeff = sum(c.N_coeff for c in controls)
pcof = 0.01 .* (2 .* rand(MersenneTwister(2026), N_coeff) .- 1)

# Control values p_k(t), q_k(t), evaluated by QGD for all three packages
function control_value(k, t, which::Symbol)
    local_pcof = QuantumGateDesign.get_control_vector_slice(pcof, controls, k)
    which === :p ? eval_p(controls[k], t, local_pcof) : eval_q(controls[k], t, local_pcof)
end
p(k) = t -> control_value(k, t, :p)
q(k) = t -> control_value(k, t, :q)

#==============================================================================
QuantumGateDesign.jl: Hermite method at order 8, checked for self-convergence
by halving the time step.
==============================================================================#
function qgd_final_states(nsteps; order=8)
    prob = qgd_problem(nsteps)
    history = eval_forward(prob, controls, pcof; order)
    return history[:, end, :] # (N_levels, N_initial_states), complex
end

prob = qgd_problem(1) # only used for initial states and its Hamiltonian matrices
U0 = complex.(prob.u0, prob.v0)
H_qgd_sys = prob.system_sym .+ im .* prob.system_asym

#==============================================================================
QuantumOptics.jl. QO's tensor(A, B) stores kron(B, A), so the bases are passed
in reverse order to get the same state layout as QGD's kron(op1, op2, op3).
==============================================================================#
qo_bases = [QO.FockBasis(n - 1) for n in subsystem_sizes]
qo_full = QO.tensor(reverse(qo_bases)...)
qo_destroy(k) = QO.embed(qo_full, length(qo_bases) + 1 - k, QO.destroy(qo_bases[k]))

qo_a = [qo_destroy(k) for k in 1:3]
qo_n = [QO.dagger(a) * a for a in qo_a]
qo_Hsys = sum(2pi * (transition_freqs[k] - rotation_freqs[k]) * qo_n[k] for k in 1:3) -
          sum(2pi * 0.5 * kerr_coeffs[k, k] * QO.dagger(qo_a[k])^2 * qo_a[k]^2 for k in 1:3) -
          sum(2pi * kerr_coeffs[j, k] * qo_n[j] * qo_n[k] for k in 1:3 for j in (k+1):3)
qo_P = [qo_a[k] + QO.dagger(qo_a[k]) for k in 1:3]
qo_Q = [im * (qo_a[k] - QO.dagger(qo_a[k])) for k in 1:3]

function qo_final_states(; tol=1e-14)
    H(t, _) = qo_Hsys + sum(p(k)(t) * qo_P[k] + q(k)(t) * qo_Q[k] for k in 1:3)
    out = similar(U0)
    for j in axes(U0, 2)
        psi0 = QO.Ket(qo_full, U0[:, j])
        _, psis = QO.timeevolution.schroedinger_dynamic([0.0, Tmax], psi0, H;
                                                        abstol=tol, reltol=tol)
        out[:, j] = psis[end].data
    end
    return out
end

#==============================================================================
QuTiP. qutip.tensor(A, B, C) is kron(A, B, C), the same layout as QGD.
==============================================================================#
qt = pyimport("qutip")
np = pyimport("numpy")
py_coefficient = pyeval("lambda f: (lambda t: float(f(t)))", Main)

function qt_destroy(k)
    ops = [k == j ? qt.destroy(n) : qt.qeye(n) for (j, n) in enumerate(subsystem_sizes)]
    return qt.tensor(ops...)
end
qt_a = [qt_destroy(k) for k in 1:3]
qt_n = [a.dag() * a for a in qt_a]
qt_Hsys = sum(2pi * (transition_freqs[k] - rotation_freqs[k]) * qt_n[k] for k in 1:3) -
          sum(2pi * 0.5 * kerr_coeffs[k, k] * qt_a[k].dag() * qt_a[k].dag() * qt_a[k] * qt_a[k] for k in 1:3) -
          sum(2pi * kerr_coeffs[j, k] * qt_n[j] * qt_n[k] for k in 1:3 for j in (k+1):3)
qt_P = [qt_a[k] + qt_a[k].dag() for k in 1:3]
qt_Q = [1im * (qt_a[k] - qt_a[k].dag()) for k in 1:3]

qt_matrix(op) = pyconvert(Matrix{ComplexF64}, op.full())

function qt_final_states(; tol=1e-12)
    H = pylist([qt_Hsys])
    for k in 1:3
        H.append(pylist([qt_P[k], py_coefficient(p(k))]))
        H.append(pylist([qt_Q[k], py_coefficient(q(k))]))
    end
    options = pydict(Dict("atol" => tol, "rtol" => tol, "nsteps" => 10^7, "store_final_state" => true,
                          "method" => "vern9"))
    dims = pylist([pylist(collect(subsystem_sizes)), pylist(ones(Int, 3))])
    out = similar(U0)
    for j in axes(U0, 2)
        psi0 = qt.Qobj(np.array(U0[:, j]); dims)
        result = qt.sesolve(H, psi0, np.array([0.0, Tmax]); options)
        out[:, j] = pyconvert(Vector{ComplexF64}, result.final_state.full().ravel())
    end
    return out
end

#==============================================================================
Comparisons
==============================================================================#
# Largest 2-norm difference over the four propagated basis states
state_error(A, B) = maximum(norm(A[:, j] - B[:, j]) for j in axes(A, 2))

println("Hamiltonian construction (max entry difference from QGD):")
@printf("  QuantumOptics.jl drift: %.2e\n", maximum(abs, Matrix(qo_Hsys.data) - H_qgd_sys))
@printf("  QuTiP drift:            %.2e\n", maximum(abs, qt_matrix(qt_Hsys) - H_qgd_sys))
for k in 1:3
    P_qgd = prob.sym_operators[k]
    Q_qgd = im .* prob.asym_operators[k]
    @printf("  control %d (QO, QuTiP): %.2e, %.2e\n", k,
            max(maximum(abs, Matrix(qo_P[k].data) - P_qgd), maximum(abs, Matrix(qo_Q[k].data) - Q_qgd)),
            max(maximum(abs, qt_matrix(qt_P[k]) - P_qgd), maximum(abs, qt_matrix(qt_Q[k]) - Q_qgd)))
end

nsteps = 4000
println("\nPropagating the four essential basis states over $(Tmax) ns ...")
U_qgd = qgd_final_states(nsteps)
U_qgd_fine = qgd_final_states(2nsteps)
U_qo = qo_final_states()
U_qt = qt_final_states()

println("\nFinal-state differences (max over the four initial states, 2-norm):")
@printf("  QGD self-convergence (nsteps %d vs %d): %.2e\n", nsteps, 2nsteps, state_error(U_qgd, U_qgd_fine))
@printf("  QGD vs QuantumOptics.jl:               %.2e\n", state_error(U_qgd_fine, U_qo))
@printf("  QGD vs QuTiP:                          %.2e\n", state_error(U_qgd_fine, U_qt))
@printf("  QuantumOptics.jl vs QuTiP:             %.2e\n", state_error(U_qo, U_qt))
