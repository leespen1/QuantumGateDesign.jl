
#=
mutable struct HermiteCarrierControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_points::Int64
    N_derivatives::Int64
    dt::Float64
    Hmat::Matrix{Float64}
    tn::Float64
    tnp1::Float64
    t_current::Float64
    fn_vals_p::Vector{Float64}
    fnp1_vals_p::Vector{Float64}
    fvals_collected_p::Vector{Float64}
    fn_vals_q::Vector{Float64}
    fnp1_vals_q::Vector{Float64}
    fvals_collected_q::Vector{Float64}
    uint_p::Vector{Float64}
    uint_q::Vector{Float64}
    uint_intermediate_p::Vector{Float64}
    uint_intermediate_q::Vector{Float64}
    ploc::Vector{Float64} # Just used for storage/working array. Values not important
    pcof_temp::Vector{Float64}
    function HermiteControl(N_points::Int64, tf::Float64, N_derivatives::Int64)
        @assert N_points > 1

        N_coeff = N_points*(N_derivatives+1)*2
        dt = tf / (N_points-1)
        Hmat = zeros(1+2*N_derivatives+1, 1+2*N_derivatives+1)
        # Hermite interpolant will be centered about the middle of each "control timestep"
        xl = 0.0
        xr = 1.0
        xc = 0.5
        icase = 0
        Hermite_map!(Hmat, N_derivatives, xl, xr, xc, icase)

        # Dummy time vals
        tn = NaN
        tnp1 = NaN
        t_current = NaN

        fn_vals_p = zeros(1+N_derivatives)
        fnp1_vals_p = zeros(1+N_derivatives)
        fvals_collected_p = zeros(2*N_derivatives + 2)

        fn_vals_q = zeros(1+N_derivatives)
        fnp1_vals_q = zeros(1+N_derivatives)
        fvals_collected_q = zeros(2*N_derivatives + 2)

        uint_p = zeros(2*N_derivatives + 2)
        uint_q = zeros(2*N_derivatives + 2)
        uint_intermediate_p = zeros(2*N_derivatives + 2)
        uint_intermediate_q = zeros(2*N_derivatives + 2)
        ploc = zeros(2*N_derivatives + 2) # Just used for storage/working array. Values not important
        
        pcof_temp = zeros(N_coeff)


        new(N_coeff, tf, N_points, N_derivatives, dt, Hmat, tn, tnp1, t_current,
            fn_vals_p, fnp1_vals_p, fvals_collected_p, 
            fn_vals_q, fnp1_vals_q, fvals_collected_q, 
            uint_p, uint_q, uint_intermediate_p, uint_intermediate_q, ploc,
            pcof_temp)
    end
end
=#



"""
Given vectors of derivatives for x and y (in the usual 1/j! format), compute
the derivatives of x*y.
"""
function product_rule!(x_derivatives, y_derivatives, prod_derivatives, N_derivatives=missing)
    if ismissing(N_derivatives)
        N_derivatives = length(prod_derivatives)-1
    end

    prod_derivatives[1:1+N_derivatives] .= 0
    # If there are extra entries in the results vector, set them to NaN
    prod_derivatives[1+N_derivatives+1:end] .= NaN 

    for dn in 0:N_derivatives # dn is the derivative order we want to compute
        for k in 0:dn
            prod_derivatives[1+dn] += x_derivatives[1+k]*y_derivatives[1+dn-k]
        end
    end
    
    return prod_derivatives
end



