using DelimitedFiles, Printf, Format

in_fname = "unitarydeviation_vs_order_nsteps.dlm"
data_full = readdlm(in_fname)

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
    if x > 0 && isinteger(log2(x))
        return "\$2^{" * int_str(log2(x)) * "}\$"
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

data_err = data_full[begin+1:end,begin+1:end-1]
data_err ./= 2 # Normalize by square root of the number of initial conditions
data_cvg = log2.(data_err[begin:end-1,:] ./ data_err[begin+1:end,:])
data_cvg = vcat(fill(NaN, 1, size(data_cvg, 2)), data_cvg)
data_err_cvg = interleave_columns(data_err, data_cvg)
data_err_cvg = hcat(data_full[begin+1:end,1], data_err_cvg)


println("Error data\n")
genericformat.(data_err) |> print_latex_table
println("\n\nConvergence data")
genericformat.(data_cvg) |> print_latex_table
println("\n\nError and Convergence Data\n")
genericformat.(data_err_cvg) |> print_latex_table


