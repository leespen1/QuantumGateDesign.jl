const fortrain_lib_str = Base.Filesystem.dirname(pathof(@__MODULE__)) * "/Controls/bspline_lib.so"

"""
Considering adding a "current_t", which would check if the t is the current t before
evaluating. Because the fortran subroutines involve only the basis functions, the

Also, because Fortran is also column major, I think I can det away with
providing a large output array, which will only be filled partially if I
evaluate fewer derivatives than necessary.

The evaluations will be done for the "clamped" bspline. The end knots are
considered repeated by fortran, we don't have to repeat them ourselves.

A clamped bspline with order k and N knots should have N-k basis functions
"""
struct FortranBSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    N_knots::Int64
    degree::Int64
    bspline_order::Int64
    knot_vector::Vector{Float64}
    work_array::Matrix{Float64}
    output_array::Matrix{Float64}
    function FortranBSplineControl(degree::Integer, tf::Real, N_knots::Integer)
        degree = convert(Int64, degree)
        tf = convert(Float64, tf)
        N_knots = convert(Int64, N_knots)

        order = degree+1
        N_basis_functions = N_knots + order - 2 #(N_nonrepeating_knots + (order-1) + (order-1) - order)
        N_coeff = 2*N_basis_functions

        # Make the knots on the interval [0,1], then scale the inputs t when evaluating
        knot_vector = Vector(LinRange(0, 1, N_knots))
        work_array = zeros(order, order)
        output_array = zeros(order, 20) # just make it large enough, see if that works

        new(N_coeff, tf, N_knots, degree, order, knot_vector, work_array, output_array)
    end
end

"""
Could do a check for t_scaled = t_current, but first let's see if this is fast or not 

For now, let's just check that results agree with BasicBSplineControl

In general, shouldn't call this this way, since it repeats lower order derivative
computations. Better to use fill_p_vec!
"""
function eval_p_derivative(control::FortranBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)
    
    # Get the index of the knot at the left of this interval
    # (which is also the index of the leftmost basis function with support in
    # this interval)
    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)

    val = 0.0
    for i in 0:control.bspline_order-1
        val += pcof[left+i] * control.output_array[1+i,1+order]
    end
    return val
end

function fill_p_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = length(vals_vec)  
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)

    for derivative_order in 0:nderiv-1
        val = 0.0
        for i in 0:control.bspline_order-1
            val += pcof[left+i] * control.output_array[1+i,1+derivative_order]
        end
        vals_vec[1+derivative_order] = val
    end
    return vals_vec
end
    
function eval_q_derivative(control::FortranBSplineControl, t::Real, pcof::AbstractVector{<: Real}, order::Int64)
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)
    
    # Get the index of the knot at the left of this interval
    # (which is also the index of the leftmost basis function with support in
    # this interval)
    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)
    offset = div(control.N_coeff, 2)

    val = 0.0
    for i in 0:control.bspline_order-1
        val += pcof[offset+left+i] * control.output_array[1+i,1+order]
    end
    return val
end

function fill_q_vec!(
        vals_vec::AbstractVector{<: Real}, control::FortranBSplineControl,
        t::Real, pcof::AbstractVector{<: Real}
    )
    # calculate derivatives up to (but not including) nderiv
    nderiv = length(vals_vec)  
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, nderiv)

    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)
    offset = div(control.N_coeff, 2)

    for derivative_order in 0:nderiv-1
        val = 0.0
        for i in 0:control.bspline_order-1
            val += pcof[offset+left+i] * control.output_array[1+i,1+derivative_order]
        end
        vals_vec[1+derivative_order] = val
    end
    return vals_vec
end

function eval_grad_p_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    # calculate derivatives up to (but not including) nderiv
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)

    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)

    # Control is linear in the pcof coefficients
    grad .= 0
    for i in 0:control.bspline_order-1
        grad[left+i] = control.output_array[1+i,1+order]
    end

    return grad
end

function eval_grad_q_derivative!(
        grad::AbstractVector{Float64}, control::FortranBSplineControl, t::Real,
        pcof::AbstractVector{<: Real}, order::Integer
    )
    t_scaled::Float64 = t / control.tf
    bsplvd!(control, t_scaled, order+1)

    left = floor(Int64, t_scaled*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)

    offset = div(control.N_coeff, 2)

    # Control is linear in the pcof coefficients
    grad .= 0
    for i in 0:control.bspline_order-1
        grad[offset+left+i] = control.output_array[1+i,1+order]
    end

    return grad
end


"""
Bezier degree elevations: 
    https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node13.html
B-spline degree elevation description: 
    https://pages.mtu.edu/~shene/COURSES/cs3621/LAB/curve/elevation.html
Knot Insertion and Removal for BSplines:
    https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node18.html

Degree elevation of a BSpline can be done by knot insertion:
1. Insert knots at internal knots until each segment of the B-spline is a bezier curve.
2. Perform degree elevation of the bezier curves (which also results in new
   control points, so pcof is updated)
3. "combining them together back to a single B-spline". Does combining them
    together mean knot removal? Because I want smoothness. So I can't have
    internal knots with multiplicity > 1. But I think the knot removal may be exact.
"""
function elevate_degree(control::FortranBSplineControl, pcof::AbstractVector{<: Real})
    
end



"""
From Netlib pppack

From  * a practical guide to splines *  by c. de Boor (7 may 92)    
calls bsplvb
calculates value and deriv.s of all b-splines which do not vanish at x

******  i n p u t  ******
  t     the knot array, of length left+k (at least)
  k     the order of the b-splines to be evaluated
  x     the point at which these values are sought
  left  an integer indicating the left endpoint of the interval of
        interest. the  k  b-splines whose support contains the interval
               (t(left), t(left+1))
        are to be considered.
  a s s u m p t i o n  - - -  it is assumed that
               t(left) .lt. t(left+1)
        division by zero will result otherwise (in  b s p l v b ).
        also, the output is as advertised only if
               t(left) .le. x .le. t(left+1) .
  nderiv   an integer indicating that values of b-splines and their
        derivatives up to but not including the  nderiv-th  are asked
        for. ( nderiv  is replaced internally by the integer  m h i g h
        in  (1,k)  closest to it.)

******  w o r k   a r e a  ******
  a     an array of order (k,k), to contain b-coeff.s of the derivat-
        ives of a certain order of the  k  b-splines of interest.

******  o u t p u t  ******
  dbiatx   an array of order (k,nderiv). its entry  (i,m)  contains
        value of  (m-1)st  derivative of  (left-k+i)-th  b-spline of
        order  k  for knot sequence  t , i=1,...,k, m=1,...,nderiv.

******  m e t h o d  ******
  values at  x  of all the relevant b-splines of order k,k-1,...,
  k+1-nderiv  are generated via  bsplvb  and stored temporarily in
  dbiatx .  then, the b-coeffs of the required derivatives of the b-
  splines of interest are generated by differencing, each from the pre-
  ceding one of lower order, and combined with the values of b-splines
  of corresponding order in  dbiatx  to produce the desired values .

"""
function bsplvd!(t::Vector{Float64}, k::Int64, x::Float64, left::Int64,
        a::Matrix{Float64}, dbiatx::Matrix{Float64}, nderiv::Int64)
    ccall(
        (:bsplvd_, fortrain_lib_str),
        Cvoid, # Return
        (Ref{Float64}, Ref{Int64}, Ref{Float64}, Ref{Int64}, Ref{Float64}, Ref{Float64}, Ref{Int64}), # Argument Types
        t, Ref(k), Ref(x), Ref(left), a, dbiatx, Ref(nderiv) # Arguments
    )
end


function bsplvd!(control::FortranBSplineControl, x::Float64, nderiv::Int64)
    # It is assumed that x ∈ [0,1]
    left::Int64 = floor(Int64, x*(control.N_knots-1) + 1)
    left = min(left, control.N_knots-1)

    bsplvd!(control.knot_vector, control.bspline_order, x, left, control.work_array, 
           control.output_array, nderiv)
end
