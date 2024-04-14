using Polynomials
# A hardcoded example
p(x) = x^7 + 1
D1p(x) = 7*x^6
D2p(x) = 7*6*x^5
D3p(x) = 7*6*5*x^4

tf = 1.0
N_control_points = 3
dt = tf / (N_control_points-1)

pcof_mat1 = zeros(4, N_control_points, 2)
for n in 1:N_control_points
    t = dt*(n-1)
    pcof_mat1[1,n,1] = p(t)
    pcof_mat1[2,n,1] = D1p(t)
    pcof_mat1[3,n,1] = D2p(t)
    pcof_mat1[4,n,1] = D3p(t)
end

# A random example
P = Polynomial(rand(7))
D1P = derivative(P, 1)
D2P = derivative(P, 2)
D3P = derivative(P, 3)
P2 = Polynomial(rand(7))
D1P2 = derivative(P2, 1)
D2P2 = derivative(P2, 2)
D3P2 = derivative(P2, 3)
pcof_mat2 = zeros(4, N_control_points, 2)
pcof_mat3 = zeros(4, N_control_points, 2)
for n in 1:N_control_points
    t = dt*(n-1)
    pcof_mat2[1,n,1] = P(t)
    pcof_mat2[2,n,1] = D1P(t)
    pcof_mat2[3,n,1] = D2P(t)
    pcof_mat2[4,n,1] = D3P(t)

    pcof_mat3[1,n,1] = P(t)
    pcof_mat3[2,n,1] = D1P(t)
    pcof_mat3[3,n,1] = D2P(t)
    pcof_mat3[4,n,1] = D3P(t)
end

pcof_mat3[1,end,1] = P2(tf)
pcof_mat3[2,end,1] = D1P2(tf)
pcof_mat3[3,end,1] = D2P2(tf)
pcof_mat3[4,end,1] = D3P2(tf)

