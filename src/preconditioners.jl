"""
Abstract supertype for preconditioners used in the forward evolution and adjoint
evolution in the discrete adjoint method.

This will be used as the left preconditioner in GMRES as implemented by the
`IterativeSolvers` package. Consequently, for a concrete subtype P the
following operations must be defined:
- `ldiv!(y, P, x)`
- `ldiv!(P, x)`
- `P \\ x`.

By default, it is assumed that an `AbstractQGDPreconditioner` has a parameter
`P` which has these operations implemented (i.e. the type simply wraps another
type which can be used as a preconditioner by `IterativeSolvers`).

We must also define the constructor P(prob::SchrodingerProb, order, adjoint),
which will be called each time a forward simulation or gradient calculation is
performed to construct the preconditioners used for the forward and adjoint
linear solves.

This is done so that the preconditioner can easily be changed as the problem
parameters and order of the method change.
"""
abstract type AbstractQGDPreconditioner end

LinearAlgebra.ldiv!(P::AbstractQGDPreconditioner, x) = ldiv!(P.P, x)
LinearAlgebra.ldiv!(y, P::AbstractQGDPreconditioner, x) = ldiv!(y, P.P, x)
Base.:\(P::AbstractQGDPreconditioner, b) = Base.:\(P.P, b)



"""
Wrapper for the IterativeSolvers.Identity preconditioner.
"""
struct IdentityPreconditioner <: AbstractQGDPreconditioner
    P::IterativeSolvers.Identity
    IdentityPreconditioner() = new(IterativeSolvers.Identity())
end

IdentityPreconditioner(prob, order, adjoint=false) = IdentityPreconditioner()



struct LUPreconditioner{T} <: AbstractQGDPreconditioner
    P::T
    function LUPreconditioner(A)
        P = LinearAlgebra.lu(A)
        T = typeof(P)
        new{T}(P)
    end
end

function LUPreconditioner(prob, order, adjoint=false)
    return LUPreconditioner(form_LHS_no_control(prob, order, adjoint))
end


"""
Preconditioner which solves the linear system exactly when the system
hamiltonian is diagonal and the control amplitudes are zero.

Assumes the hamiltonian is diagonal and solves using Gaussian elimination.
"""
struct DiagonalHamiltonianPreconditioner <: AbstractQGDPreconditioner
    LHS_diagonal::Vector{Float64} 
    LHS_upper_diagonal::Vector{Float64}
    LHS_lower_diagonal::Vector{Float64}
    real_system_size::Int64
    complex_system_size::Int64
end



function DiagonalHamiltonianPreconditioner(prob, order, adjoint=false)
    return DiagonalHamiltonianPreconditioner(form_LHS_no_control(prob, order, adjoint))
end


"""
Construct preconditioner based on arbitrary LHS matrix.
E.g. preconditioner for LHS*x = b
"""
function DiagonalHamiltonianPreconditioner(LHS::AbstractMatrix)
    @assert size(LHS, 1) == size(LHS, 2)
    @assert iseven(size(LHS, 1))
    real_system_size = size(LHS, 1)
    complex_system_size = div(real_system_size, 2)

    LHS_diagonal = LinearAlgebra.diag(LHS)
    @assert all(LHS_diagonal .!= 0) # Make sure no diagonal entries are zero

    LHS_upper_diagonal = LinearAlgebra.diag(LHS, complex_system_size)
    LHS_lower_diagonal = LinearAlgebra.diag(LHS, -complex_system_size)

    return DiagonalHamiltonianPreconditioner(
        LHS_diagonal, LHS_upper_diagonal, LHS_lower_diagonal,
        real_system_size, complex_system_size
    )
end


function LinearAlgebra.ldiv!(y::AbstractVector, P::DiagonalHamiltonianPreconditioner, x::AbstractVector)
    y .= x
    ldiv!(P, y)
end


function LinearAlgebra.ldiv!(P::DiagonalHamiltonianPreconditioner, x::AbstractVector)
    N = P.complex_system_size
    # Should be able to do this very broadcasted
    for i in eachindex(P.LHS_lower_diagonal)
        # Clear lower-left block diagonal
        ratio = P.LHS_lower_diagonal[i] / P.LHS_diagonal[i]
        x[N+i] -= x[i]*ratio
        # Get bottom-right block diagonal to ones
        x[N+i] /= (P.LHS_diagonal[N+i] - P.LHS_upper_diagonal[i]*ratio)
    end
    for i in eachindex(P.LHS_lower_diagonal)
        # Clear diagonal of of upper-right block
        x[i] -= P.LHS_upper_diagonal[i] * x[N+i]
        # Get upper-left block diagonal to ones
        x[i] /= P.LHS_diagonal[i]
    end
      
    return x
end

function Base.:\(P::DiagonalHamiltonianPreconditioner, b::AbstractVector)
    x = similar(b)
    ldiv!(x, P, b)
end

function lu_preconditioner(prob, order, adjoint=false)
    LHS = form_LHS_no_control(prob, order, adjoint)
    preconditioner = lu(LHS)
    return preconditioner
end
