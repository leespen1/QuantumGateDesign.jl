using QuantumGateDesign, LinearAlgebra, PrettyTables, Printf, Format

Ω = 0.05
#Ω = 0.5
θ = angle(Ω)
Ωabs = abs(Ω)
p0 = real(Ω)
q0 = imag(Ω)
N_rabi_oscillations = 10
T = 10pi/abs(Ω) # 10 rabi oscillations
#T = 10pi/(2*abs(Ω)) # 10 rabi oscillations

function U(t)
    Umat = zeros(ComplexF64, 2, 2)
    Umat[1,1] = cos(abs(Ω)*t)
    Umat[2,1] = -(sin(θ)+im*cos(θ))*sin(abs(Ω)*t)
    Umat[1,2] =  (sin(θ)-im*cos(θ))*sin(abs(Ω)*t)
    #Umat[2,1] = sin(abs(Ω)*t)
    #Umat[1,2] = sin(abs(Ω)*t)
    Umat[2,2] = cos(abs(Ω)*t)
    return Umat
end


prob = QuantumGateDesign.rabi_oscillator_problem(tf=T, gmres_abstol=1e-10, gmres_reltol=1e-10, nsteps=2)
control = QuantumGateDesign.GRAPEControl(1, prob.tf)

function collect_data(iter_range, orders, norm_type=Inf, error_type=:final_time)
    nsteps_vec = fill(NaN, length(iter_range))
    data_err = fill(NaN, length(iter_range), length(orders))
    data_cvg = fill(NaN, length(iter_range), length(orders))
    for (i,nsteps_exp) in enumerate(iter_range)
        prob.nsteps = 2^nsteps_exp
        nsteps_vec[i] = prob.nsteps
        for (k, order) in enumerate(orders)
            history_numerical = eval_forward(prob, control, pcof, order = order)

            ts = LinRange(0, T, 1+prob.nsteps)
            history_analytic = cat(U.(ts)..., dims=3)
            history_analytic = permutedims(history_analytic, (1,3,2)) # Use my index ordering

            if error_type == :final_time
                numerical_sol = history_numerical[:,end,:]
                analytic_sol = history_analytic[:,end,:]
            elseif error_type == :full_time
                numerical_sol = history_numerical
                analytic_sol = history_analytic
            else
                throw(ArgumentError(error_type))
            end
            error = numerical_sol - analytic_sol

            # Don't use relative error for L∞, I don't think that makes sense
            if norm_type == Inf 
                error_norm = norm(error, norm_type)
            else
                error_norm = norm(error, norm_type) / norm(analytic_sol)
            end
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
    if isnan(x)
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

pcof = [p0, q0]
orders = [2,4,6,8,10,12]
#orders = [2,4,6]
iter_range = 4:8
header = vcat("# Steps", ["Order $order" for order in orders])
nsteps_vec, data_err, data_cvg = collect_data(iter_range, orders, 2)

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
