using QuantumGateDesign, LinearAlgebra, PrettyTables, Printf, Format, Zygote

# User parameters
N_rabi_oscillations = 9.5
θpi = 1/4
r = 0.05
Utarg = [1 1;1 -1] ./ sqrt(2)

# Derived variables
Ω = r*cispi(θpi)
p = real(Ω)
q = imag(Ω)
s(t) = sinpi(r*t)
c(t) = cospi(r*t)
T = N_rabi_oscillations/r 


function U(t, Ω)
    θ = angle(Ω)
    Umat11 = cospi(abs(Ω)*t)
    Umat21 = -(sin(θ)+im*cos(θ))*sinpi(abs(Ω)*t)
    Umat12 =  (sin(θ)-im*cos(θ))*sinpi(abs(Ω)*t)
    Umat22 = cospi(abs(Ω)*t)
    return [Umat11 Umat12; Umat21 Umat22]
end

function infidelity(targ, UT)
    @assert size(targ) == size(UT)
    E = size(UT, 2)
    val = 1 - (1/E^2)*abs(dot(targ, UT))^2
    return val
end

function infidelity_grad_fd(targ, Ω)
    d = 1e-8
    grad1 = (infidelity(targ, U(T,Ω+d)) - infidelity(targ, U(T,Ω-d)))/(2d)
    grad2 = (infidelity(targ, U(T,Ω+im*d)) - infidelity(targ, U(T,Ω-im*d)))/(2d)
    return [grad1, grad2]
end

function infidelity_grad_AD(targ, Ω)
    return gradient(x -> infidelity(targ, U(T,x)), Ω)
end

function dUdp(t)
    # Need t*pi instaed of just t outside of sin/cos, since I am giving t in
    # units of 1/pi, which is accoutned for by use of sinpi
    # # may need t -> pi*t
    U11 = -s(t)*p*t*pi/r
    U12 = -( (q/r)*s(t) + im*p*t*pi*c(t) ) / (p-im*q)
    U21 =  ( (q/r)*s(t) - im*p*t*pi*c(t) ) / (p+im*q)
    U22 = U11
    return [U11 U12; U21 U22]
end

function dUdq(t)
    # Need t*pi instaed of just t outside of sin/cos, since I am giving t in
    # units of 1/pi, which is accoutned for by use of sinpi
    # # may need t -> pi*t
    U11 = -s(t)*q*t*pi/r
    U12 =  ( q*t*pi*c(t) + (im*p/r)*s(t) ) / (q+im*p)
    U21 = -( (p/r)*s(t) + im*q*t*pi*c(t) ) / (p+im*q)
    U22 = U11
    return [U11 U12; U21 U22]
end

function partialInfidelity(UT, UT_partial)
    E = size(UT, 2)
    f1 = dot(real(Utarg), real(UT)) + dot(imag(Utarg), imag(UT))
    f2 = dot(real(Utarg), real(UT_partial)) + dot(imag(Utarg), imag(UT_partial))
    f3 = dot(real(Utarg), imag(UT)) - dot(imag(Utarg), real(UT))
    f4 = dot(real(Utarg), imag(UT_partial)) - dot(imag(Utarg), real(UT_partial))
    return -(2/E^2)*(f1*f2 + f3*f4)
end

prob = QuantumGateDesign.rabi_oscillator_problem(tf=T*pi, gmres_abstol=1e-15, gmres_reltol=1e-15, nsteps=2)
control = QuantumGateDesign.GRAPEControl(1, prob.tf)

function collect_data(iter_range, orders)
    nsteps_vec = fill(NaN, length(iter_range))
    data_err = fill(NaN, length(iter_range), length(orders))
    data_cvg = fill(NaN, length(iter_range), length(orders))

    for (i,nsteps_exp) in enumerate(iter_range)
        prob.nsteps = 2^nsteps_exp
        nsteps_vec[i] = prob.nsteps
        for (k, order) in enumerate(orders)
            history_numerical = eval_forward(prob, control, pcof, order = order)
            numerical_sol = history_numerical[:,end,:]
            analytic_sol = U(T, Ω)

            error = numerical_sol - analytic_sol
            error_norm = norm(error, 2) / max(norm(analytic_sol), 1e-15)

            data_err[i, k] = error_norm
            if i > 1
                data_cvg[i, k] = abs(log10(data_err[i,k]/data_err[i-1,k])/log10(nsteps_vec[i]/nsteps_vec[i-1]))
            end
        end
    end
    return nsteps_vec, data_err, data_cvg
end

function collect_data2(iter_range, orders)
    nsteps_vec = fill(NaN, length(iter_range))
    data_err = fill(NaN, length(iter_range), length(orders))
    data_cvg = fill(NaN, length(iter_range), length(orders))

    UT_dp = dUdp(T)
    UT_dq = dUdq(T)
    UT = U(T, Ω)
    infidelity_partial_p =  partialInfidelity(UT, UT_dp)
    infidelity_partial_q =  partialInfidelity(UT, UT_dq)
    analytic_grad =  [infidelity_partial_p, infidelity_partial_q]
    #analytic_fd_grad = infidelity_grad_fd(Utarg, Ω)
    analytic_AD_grad_complex = infidelity_grad_AD(Utarg, Ω)[1]
    analytic_AD_grad = [real(analytic_AD_grad_complex), imag(analytic_AD_grad_complex)]

    for (i,nsteps_exp) in enumerate(iter_range)
        prob.nsteps = 2^nsteps_exp
        nsteps_vec[i] = prob.nsteps
        for (k, order) in enumerate(orders)
            numerical_grad = discrete_adjoint(prob, control, pcof, Utarg, order = order)
            numerical_grad = numerical_grad
            error = numerical_grad - analytic_grad
            #error = numerical_grad - analytic_AD_grad

            ## TODO Make error relative, with a max statement to prevent roundoff issues
            error_norm = norm(error, 2) / max(norm(analytic_AD_grad), 1e-15)
            data_err[i, k] = error_norm
            if i > 1
                data_cvg[i, k] = abs(log10(data_err[i,k]/data_err[i-1,k])/log10(nsteps_vec[i]/nsteps_vec[i-1]))
            end
        end
    end
    return nsteps_vec, data_err, data_cvg
end



function interleave_columns(A, B)
    @assert size(A) == size(B) "Matrices must be the same size"
    m, n = size(A)
    C = similar(A, m, 2n)
    C[:, 1:2:2n] .= A
    C[:, 2:2:2n] .= B
    return C
end

"""
Convert a floating point number to scientific format, e.g. 1.7(-1) for 0.17.
If the exponent is zero, omit it.
If the number is an integer, print it as an integer
"""
function sci_str(x::Real; digits=2)
    if isnan(x) || isinf(x)
        return "-"
    end

    str = @sprintf("%.*e", digits-1, x)  # e.g., "1.700e-01"
    base, exp = split(str, 'e')
    exp = parse(Int, exp)

    if exp == 0
        return "$(base)"
    else
        return "$(base)($(exp))"
    end
end

function int_str(x::Real)
    return format(Int(x), commas=true)
end

function nonsci_str(x::Real; digits=2)
    if isnan(x)
        return "-"
    end

    return @sprintf("%.*f", digits-1, x)  # e.g., "1.700e-01"
end

function genericformat(x::Real)
    if isinteger(x)
        return int_str(x)
    end
    return sci_str(x)
end

function print_latex_table(A)
    for row in eachrow(A)
        row_str = reduce((a,b) -> a*" & "*b, row)
        row_str = row_str * " \\\\"
        println(row_str)
    end
end

pcof = [p, q]
orders = [2,4,6,8,10,12]
#orders = [2,4,6]
iter_range = 4:8
header = vcat("# Steps", ["Order $order" for order in orders])



println("Numerical Solution Accuracy")
nsteps_vec, data_err, data_cvg = collect_data(iter_range, orders)

nsteps_strs = int_str.(nsteps_vec)
data_err_strs = sci_str.(data_err)
data_cvg_strs = nonsci_str.(data_cvg)

table_err_strs = hcat(nsteps_strs, data_err_strs)
table_cvg_strs = hcat(nsteps_strs, data_cvg_strs)
table_comb_strs = hcat(nsteps_strs, interleave_columns(data_err_strs, data_cvg_strs))

pretty_table(table_err_strs, header=header)
pretty_table(table_cvg_strs, header=header)
#pretty_table(table_comb_strs)
print_latex_table(table_comb_strs)


println("Gradient Accuracy")
nsteps_vec, data_err, data_cvg = collect_data2(iter_range, orders)

nsteps_strs = int_str.(nsteps_vec)
data_err_strs = sci_str.(data_err)
data_cvg_strs = nonsci_str.(data_cvg)

table_err_strs = hcat(nsteps_strs, data_err_strs)
table_cvg_strs = hcat(nsteps_strs, data_cvg_strs)
table_comb_strs = hcat(nsteps_strs, interleave_columns(data_err_strs, data_cvg_strs))

pretty_table(table_err_strs, header=header)
pretty_table(table_cvg_strs, header=header)
#pretty_table(table_comb_strs)
print_latex_table(table_comb_strs)

println("UT = ")
display(U(T, Ω))
println("UT (numerical, nsteps=$(prob.nsteps)) = ")
display(eval_forward(prob, control, pcof, order=12)[:,end,:])
println("Utarg = ")
display(Utarg)
println("Infidelity = ", infidelity(Utarg, U(T,Ω)))

UT_dp = dUdp(T)
UT_dq = dUdq(T)
UT = U(T, Ω)
infidelity_partial_p =  partialInfidelity(UT, UT_dp)
infidelity_partial_q =  partialInfidelity(UT, UT_dq)
analytic_grad =  [infidelity_partial_p, infidelity_partial_q]
println("Analytic gradient = ")
display(analytic_grad)
