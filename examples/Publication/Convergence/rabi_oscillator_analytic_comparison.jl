using QuantumGateDesign
using LinearAlgebra
using PrettyTables
using Printf
using LaTeXStrings
using LaTeXTabulars

"""
Analytic solution operator, where p₀ = Re(Ω), q₀ = Im(Ω).
"""
function U(t, Ω)
    θ = angle(Ω)
    Umat = zeros(ComplexF64, 2, 2)
    Umat[1,1] = cos(abs(Ω)*t)
    Umat[2,1] = -(sin(θ)+im*cos(θ))*sin(abs(Ω)*t)
    Umat[1,2] =  (sin(θ)-im*cos(θ))*sin(abs(Ω)*t)
    #Umat[2,1] = sin(abs(Ω)*t)
    #Umat[1,2] = sin(abs(Ω)*t)
    Umat[2,2] = cos(abs(Ω)*t)
    return Umat
end

H0_sym = zeros(2,2)
H0_asym = zeros(2,2)
a = [0.0 1;
     0   0]
Hc_sym = a + a'
Hc_asym = a - a'

sym_ops = [Hc_sym]
asym_ops = [Hc_asym]

U0_complex = [1.0 0;
              0   1]

u0 = real(U0_complex)
v0 = imag(U0_complex)

Ω = 0.5
H_complex = Ω*a + conj(Ω)*a'

# 5 Rabi Oscillations
tf = 10pi / (2*abs(Ω))
nsteps = 10
N_ess_levels = 2

prob = SchrodingerProb(
    H0_sym, H0_asym, sym_ops, asym_ops, u0, v0, tf, nsteps, N_ess_levels
)

N_control_coeff = 1
control = QuantumGateDesign.GRAPEControl(N_control_coeff, prob.tf)

pcof = [real(Ω), imag(Ω)]

history_numerical = eval_forward(prob, control, pcof, order=4)
history_numerical = history_numerical[1:2, 1, :, :]  + im*history_numerical[3:4, 1, :, :]
#history_numerical = real_to_complex(history_numerical[:,1,:,:])

history_analytic = similar(history_numerical)
history_exp = similar(history_numerical)
for (i, t) in enumerate(ts)
    history_analytic[:,i,:] .= U(t,Ω)*U0_complex
    history_exp[:,i,:] .= exp(-im*H_complex*t)*U0_complex
end

orders = [2,4,6,8]
nsteps_iter = 5 .* (2 .^ (0:10))
header = vcat([raw"\# Steps", L"\Delta t"], orders)
data = Matrix{Any}(undef, length(nsteps_iter), length(header))
for (i, order) in enumerate(orders)
    for (j, nsteps) in enumerate(nsteps_iter)
        prob.nsteps = nsteps

        history_numerical = eval_forward(prob, control, pcof, order=order)
        # convert to complex
        history_numerical = history_numerical[1:2, 1, :, :]  + im*history_numerical[3:4, 1, :, :]
        history_analytic = similar(history_numerical)
        ts = LinRange(0, prob.tf, 1+prob.nsteps)
        for (k, t) in enumerate(ts)
            history_analytic[:,k,:] .= U(t,Ω)#*U0_complex
        end
        errors = history_numerical - history_analytic
        error_linf = norm(errors, Inf)
        error_l2 = norm(errors, 2)

        data[j, 1] = nsteps
        data[j, 2] = prob.tf / nsteps
        data[j, 2+i] = error_linf
    end
end

results_table = pretty_table(
    data;
    header=header
)

# Format Int and Float arguments correctly as strings
formatter(x::Int) = @sprintf("%d", x)
formatter(x::Float64) = @sprintf("%.1e", x)
formatted_data = formatter.(data)

table_name = "/tmp/table.tex"
latex_tabular(
    table_name,
    Tabular("ll"*"l"^length(orders)),
    [  # Table lines
        Rule(:top), # Top rule
        # Top row, 2nd and 3rd entries each span two columns
        ["", "", MultiColumn(length(orders), :c, "Order")],
        CMidRule("lr", 3, 2+length(orders)),
        header,
        Rule(:mid), # Full horizontal rule
        formatted_data, # The data
        Rule(:bottom), # Bottom rule
    ]
)
