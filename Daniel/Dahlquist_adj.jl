using Plots
using ForwardDiff

# Here we solve y'=a*y, y(0)=1 by numerical and exact means.
# We compute the gradient of the continuous cost function
# J(a) = 0.5*(y(T,b)-y(T,a))^2
# And the cost function of the discretized solution to Dahlquist
# K(a) = 0.5*(C-y_Nt)^2

# We know the exact analytic solution 
function y_exact(a::Real,T::Real)
    return exp(a*T)
end

# And the exact analytic gradient
function gradient_exact(a::Real,b::Real,T::Real)
    return -T*exp(a*T)*(exp(b*T)-exp(a*T))
end

# Numerical solution to Dahlquist
function y_by_Euler(a::Real,dt::Real,N::Int)

    y = 1.0
    for i = 0:N-1
        y = y + dt*a*y
    end
    return y
    
end

# Z is the solution to the equation (dy/da)' = a*(dy/da)+y, (dy/da)(t=0) = 0
# We are lazy and recompute y
function z_by_Euler(a::Real,dt::Real,N::Int)

    z = 0.0
    y = 1.0
    for i = 0:N-1
        z = z + dt*a*z + dt*y
        y = y + dt*a*y
    end
    return z
    
end


T = 1.0
b = 2.0
a = 1.0
y_ex(x) = y_exact(x,T)
J(x) = 0.5*(y_exact(b,T)-y_ex(x))^2

println("Exact gradient = ",gradient_exact(a,b,T))
println("FD/AD gradient = ",ForwardDiff.derivative(J, a))

Nt = 1000
dt = T/Nt

y_by_E(x) = y_by_Euler(x,dt,Nt)
y_disc_final = y_exact(b,T)

K(x) = 0.5*(y_disc_final - y_by_E(x))^2

EDG = -(y_disc_final - y_by_E(a))*z_by_Euler(a,dt,Nt)

println("FD/AD discrete gradient = ",ForwardDiff.derivative(K, a))
println("Exact discrete gradient = ",EDG)

