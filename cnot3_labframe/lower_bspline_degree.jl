using QuantumGateDesign


"""
Given a high-degree B-Spline and control vector, use least-squares to fit a
lower-degree B-spline (with the same number of control parameters) to the
high-degree B-Spline.
"""
function lower_bspline_degree(
    high_degree::Integer, low_degree::Integer, tf::Real,
    high_deg_pcof::AbstractVector{<: Real}, points_per_basis_func::Integer=10
)
    N_basis_functions = div(length(high_deg_pcof), 2)
    @assert N_basis_functions*2 == length(high_deg_pcof) "Pcof length must be divisible by two!"

    # Build controls
    high_deg_control = FortranBSplineControl(high_degree, N_basis_functions, tf)
    low_deg_control = FortranBSplineControl(low_degree, N_basis_functions, tf)

    # Sample high degree control
    t_range = LinRange(0, tf, points_per_basis_func*N_basis_functions) 
    p_samples = [eval_p_derivative(high_deg_control, t, high_deg_pcof, 0)
                 for t in t_range]
    q_samples = [eval_q_derivative(high_deg_control, t, high_deg_pcof, 0)
                 for t in t_range]

    # Sample basis functions of low degree control
    basis_pcofs_mat = zeros(2*N_basis_functions, 2*N_basis_functions)
    for i in 1:2*N_basis_functions
        basis_pcofs_mat[i,i] = 1
    end
    basis_pcofs = basis_pcofs_mat |> eachcol |> collect

    Ap = [eval_p_derivative(low_deg_control, t, basis_pcof, 0)
          for t in t_range, basis_pcof in basis_pcofs[1:N_basis_functions]]
    Aq = [eval_q_derivative(low_deg_control, t, basis_pcof, 0)
          for t in t_range, basis_pcof in basis_pcofs[1+N_basis_functions:2*N_basis_functions]]

    # Perform least squares fit of low degree basis functions to high degree sample
    low_deg_pcof_p = Ap \ p_samples
    low_deg_pcof_q = Aq \ q_samples
    
    low_deg_pcof = vcat(low_deg_pcof_p, low_deg_pcof_q)

    # Check agreement over even finer time grid
    fine_t_range = LinRange(0, tf, points_per_basis_func*N_basis_functions)
    dt = tf / (length(fine_t_range)-1)
    low_deg_p_samples = [eval_p_derivative(low_deg_control, t, low_deg_pcof, 0)
                 for t in fine_t_range]
    low_deg_q_samples = [eval_q_derivative(low_deg_control, t, low_deg_pcof, 0)
                 for t in fine_t_range]
    low_deg_fine_samples = vcat(low_deg_p_samples, low_deg_q_samples)

    high_deg_p_samples = [eval_p_derivative(high_deg_control, t, high_deg_pcof, 0)
                 for t in fine_t_range]
    high_deg_q_samples = [eval_q_derivative(high_deg_control, t, high_deg_pcof, 0)
                 for t in fine_t_range]
    high_deg_fine_samples = vcat(high_deg_p_samples, high_deg_q_samples)

    @show maximum(abs, low_deg_fine_samples .- high_deg_fine_samples)
    L2_norm_diff = sum(abs2, low_deg_fine_samples .- high_deg_fine_samples) .* dt |> sqrt
    println("Discrete L2 norm of difference between 2 samples: ", L2_norm_diff)

    return low_deg_pcof
end
