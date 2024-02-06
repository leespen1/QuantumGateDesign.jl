#=
This code is for testing the correctness of calculating the derivatives of the
state vector. The "matrix-free" calculations used in the forward evolution and
gradient calculations are compared against hard-coded, matrix-using methods
of order 2, 4, and 6.

I.e. for the hard-coded method:

Instead of computing w'' by computing w' = Aw and then w'' = A'w + Aw',
we would compute w'' = (A' + AA)w, with no recursion.

For the adjoint "derivatives" (which are not really derivatives), we directly
take the transpose of the matrix applied 


system_sym = [1 0
              0 0]

sym_ops = [
θcos(t) *[0 1
          1 0]
]

Notes/Improvements
- This test does not include any assymetric operators
- This test only uses a single control
- This test does not include any forcing 

=#
using HermiteOptimalControl
using Test

@testset "Checking Correctness of Derivative Calculations" begin

Ks = [1.0 0; 0 0]
Kc = [0.0 1; 1 0]

K(t, θ) = Ks + θ*cos(t)*Kc
dK_dt1(t, θ)= -θ*sin(t)*Kc
dK_dt2(t, θ)= -θ*cos(t)*Kc
dK_dt3(t, θ)=  θ*sin(t)*Kc

dKdθ(t, θ) =      cos(t)*Kc
dKdθ_dt1(t, θ) = -sin(t)*Kc
dKdθ_dt2(t, θ) = -cos(t)*Kc
dKdθ_dt3(t, θ) =  sin(t)*Kc

Z = zeros(2,2)

A(t, θ) = [Z       K(t,θ)
           -K(t,θ) Z]

dA_dt1(t, θ) = [Z           dK_dt1(t,θ)
               -dK_dt1(t,θ) Z]

dA_dt2(t, θ) = [Z           dK_dt2(t,θ)
               -dK_dt2(t,θ) Z]

dA_dt3(t, θ) = [Z           dK_dt3(t,θ)
               -dK_dt3(t,θ) Z]



dAdθ(t, θ) = [Z          dKdθ(t,θ)
              -dKdθ(t,θ) Z]

dAdθ_dt1(t, θ) = [Z          dKdθ_dt1(t,θ)
              -dKdθ_dt1(t,θ) Z]

dAdθ_dt2(t, θ) = [Z          dKdθ_dt2(t,θ)
              -dKdθ_dt2(t,θ) Z]

dAdθ_dt3(t, θ) = [Z          dKdθ_dt3(t,θ)
              -dKdθ_dt3(t,θ) Z]


function calculate_derivative(u, v, t, θ)
    w = vcat(u, v) 
    result = zeros(size(w,1), 5)

    result[:,1] .= w

    result[:,2] .= A(t,θ)*w

    result[:,3] .= (dA_dt1(t,θ) + A(t,θ)^2)*w ./ factorial(2)

    result[:,4] .= (dA_dt2(t,θ) 
                    + (2*dA_dt1(t,θ)*A(t,θ))
                    + (A(t,θ)*dA_dt1(t,θ))
                    + (A(t,θ)^3)
                   )*w ./ factorial(3)

    result[:,5] .= (
        dA_dt3(t,θ)
        + 3*dA_dt2(t,θ)*A(t,θ)
        + 3*(dA_dt1(t,θ))^2
        + 3*dA_dt1(t,θ)*(A(t,θ)^2)
        + A(t,θ)*dA_dt2(t,θ)
        + 2*A(t,θ)*dA_dt1(t,θ)*A(t,θ)
        + (A(t,θ)^2)*dA_dt1(t,θ)
        + A(t,θ)^4
    )*w ./ factorial(4)


    return result
end


function calculate_adjoint_derivative(u, v, t, θ)
    w = vcat(u, v) 
    result = zeros(size(w,1), 5)

    result[:,1] .= w

    result[:,2] .= transpose(A(t,θ))*w

    result[:,3] .= transpose(dA_dt1(t,θ) + A(t,θ)^2)*w ./ factorial(2)

    result[:,4] .= transpose(
        dA_dt2(t,θ) 
        + 2*dA_dt1(t,θ)*A(t,θ) 
        + A(t,θ)*dA_dt1(t,θ)
        + A(t,θ)^3
       )*w ./ factorial(3)

    result[:,5] .= transpose(
        dA_dt3(t,θ)
        + 3*dA_dt2(t,θ)*A(t,θ)
        + 3*(dA_dt1(t,θ))^2
        + 3*dA_dt1(t,θ)*(A(t,θ)^2)
        + A(t,θ)*dA_dt2(t,θ)
        + 2*A(t,θ)*dA_dt1(t,θ)*A(t,θ)
        + (A(t,θ)^2)*dA_dt1(t,θ)
        + A(t,θ)^4
    )*w ./ factorial(4)

    return result
end


function calculate_partial_derivative(u, v, t, θ)
    w = vcat(u, v) 
    result = zeros(size(w,1), 4)

    result[:,1] .= 0

    result[:,2] .= dAdθ(t,θ)*w

    result[:,3] .= (dAdθ_dt1(t,θ) 
                    + dAdθ(t,θ)*A(t,θ) + A(t,θ)*dAdθ(t,θ)
                   )*w ./ factorial(2)

    result[:,4] .= (dAdθ_dt2(t,θ) 
                    + 2*dAdθ_dt1(t,θ)*A(t,θ) + 2*dA_dt1(t,θ)*dAdθ(t,θ)
                    + dAdθ(t,θ)*dA_dt1(t,θ) + A(t,θ)*dAdθ_dt1(t,θ)
                    + dAdθ(t,θ)*A(t,θ)^2 + A(t,θ)*dAdθ(t,θ)*A(t,θ) + (A(t,θ)^2)*dAdθ(t,θ)
                   )*w ./ factorial(3)

    return result
end

state_vector = rand(4)
t = rand()
θ = rand()


system_sym = Ks
system_asym = Z
sym_ops = [Kc]
asym_ops = [Z]
# The following 4 variables do not matter for this test
u0 = zeros(2)
v0 = zeros(2)
tf = 1.0
nsteps = 1
# The following 2 variables matter
N_ess_levels = 2
N_guard_levels = 0

prob = SchrodingerProb(system_sym, system_asym, sym_ops, asym_ops, u0, v0, tf, nsteps, N_ess_levels, N_guard_levels)
control = HermiteOptimalControl.SingleSymCosControl(tf, frequency=1)
pcof = [θ]

# Regular Derivatives Test
hard_coded_derivatives = calculate_derivative(state_vector[1:2], state_vector[3:4], t, θ)

N_derivatives = 4
package_derivatives = zeros(length(state_vector), N_derivatives+1)
package_derivatives[:,1] .= state_vector # Put state vector in first column of derivative matrix
compute_derivatives!(package_derivatives, prob, control, t, pcof, N_derivatives)

@test isapprox(package_derivatives, hard_coded_derivatives, atol=1e-15, rtol=1e-15)


# Adjoint Derivatives Test
hard_coded_adjoint_derivatives = calculate_adjoint_derivative(state_vector[1:2], state_vector[3:4], t, θ)

N_derivatives = 4
package_adjoint_derivatives = zeros(length(state_vector), N_derivatives+1)
package_adjoint_derivatives[:,1] .= state_vector # Put state vector in first column of derivative matrix
compute_adjoint_derivatives!(package_adjoint_derivatives, prob, control, t, pcof, N_derivatives)

@test isapprox(package_adjoint_derivatives, hard_coded_adjoint_derivatives, atol=1e-15, rtol=1e-15)


# Partial Derivative Test
hard_coded_partial_derivatives = calculate_partial_derivative(state_vector[1:2], state_vector[3:4], t, θ)

N_derivatives = 3
pcof_index = 1
package_partial_derivatives = zeros(length(state_vector), N_derivatives+1)
compute_partial_derivative!(package_partial_derivatives, package_derivatives, prob, control, t, pcof, N_derivatives, pcof_index)

@test isapprox(package_partial_derivatives, hard_coded_partial_derivatives, atol=1e-15, rtol=1e-15)

end #testset
