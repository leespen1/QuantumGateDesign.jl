
#=

This code will attempt to hard-code the gradient accumulation for 

system_sym = [1 0
              0 0]

sym_ops = [
θcos(t) *[0 1
          1 0]
]

=#

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

    result[:,2] .= A(t,θ)'*w

    result[:,3] .= (dA_dt1(t,θ) + A(t,θ)^2)'*w ./ factorial(2)

    result[:,4] .= (dA_dt2(t,θ) + 2*dA_dt1(t,θ)*A(t,θ) + A(t,θ)*dA_dt1(t,θ) + A(t,θ)^3)'*w ./ factorial(3)

    result[:,5] .= (
        dA_dt3(t,θ)
        + 3*dA_dt2(t,θ)*A(t,θ)
        + 3*(dA_dt1(t,θ))^2
        + 3*dA_dt1(t,θ)*(A(t,θ)^2)
        + A(t,θ)*dA_dt2(t,θ)
        + 2*A(t,θ)*dA_dt1(t,θ)*A(t,θ)
        + (A(t,θ)^2)*dA_dt1(t,θ)
        + A(t,θ)^4
    )'*w ./ factorial(4)

    return result
end


function calculate_partial_derivative(u, v, t, θ)
    w = vcat(u, v) 
    result = zeros(size(w,1), 4)

    result[:,1] .= w

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
