function Hermite_map!(tmat,m,xl,xr,xc,icase)

    #=
 ! This subroutine computes the coefficient matrix tmat which
 ! transfers derivative data of order through m at xl and xr
 ! to the function and derivative data at xc
 ! That is, if p is a polynomial of degree 2m+1
 !
 !  h^k D^k p (xc)/k! = sum_(j=0)^m tmat(k,j) h^j D^j p(xl)/j!
 !                                + tmat(k,j+m+1) h^j D^j p(xr)/j!
 !
 !  icase < 0 => xc=xl (left boundary case)
 !  icase = 0 => xl < xc < xr
 !  icase > 0 => xc=xr (right boundary case)
 IMPLICIT NONE
 INTEGER, INTENT(IN) :: m,icase
 DOUBLE PRECISION, INTENT(IN) :: xl,xr,xc
 DOUBLE PRECISION, DIMENSION(0:2*m+1,0:2*m+1), INTENT(OUT) :: tmat
 DOUBLE PRECISION, DIMENSION(0:m+1,0:m+1) :: bcofs
 DOUBLE PRECISION :: h,z,zc,adl,adr,sign,c1l,c1r,c2l,c2r
 INTEGER :: i,j,k

 ! Compute in normalized coordinates
     =#
    
    h = xr-xl
    z = (xc-xl)/h
    zc = z-1.0
    
    if (icase > 0) 
        z = 1.0
        zc = 0.0
    elseif icase < 0
        z = 0.0
        zc = -1.0
    end

    bcofs = zeros(m+2,m+2)
    binomial!(bcofs,m+1)

#=    
 !
 ! We begin with the Hermite-Lagrange interpolants:
 !
 !   Q_j (z) = z^(m+1) sum_{k=j}^m h_{kj} (z-1)^k,
 !
 !   j=0, ... , m
 !
 !   satisfying Q_j = (z-1)^j + O((z-1)^(m+1)) at z=1
 !
 !   After some algebra once can show:
 !
 !   h_jj = 1,  h_kj = -sum_{p=j}^{k-1} b_(k-p)^(m+1) h_pj ,
 !              for k>j
 !
 !   here b_(k-p)^(m+1) is the binomial coefficient (m+1)!/((k-p)!(m+1-k+p)!)
 !
 ! To construct the matrix we
 ! now evaluate the interpolant and its derivatives at z
 !
 ! Data to the left is handled by a simple change of variables z -> 1-z
 !
 ! Start with the last column - note that we directly use the recursive
 ! definition of the h's to fold previously computed columns into old
 ! ones. Note that the polynomial is now centered about the midpoint
 !
=#

    for i = 0:2*m+1
        adl = 0.0
        adr = 0.0
        for j = max(0,i-m):min(i,m+1)
            if (m-i+j) ==  0 
                c2l = 1.0
                c2r = 1.0
            else
                c2l = z^(m-i+j)
                c2r = zc^(m-i+j)
            end 
            if (m+1-j) == 0 
                c1l = 1.0
                c1r = 1.0
            else
                c1l = zc^(m+1-j)
                c1r = z^(m+1-j)
            end 
            adr = adr+bcofs[1+m+1,1+j]*bcofs[1+m,1+i-j]*c1r*c2r
            adl = adl+bcofs[1+m+1,1+j]*bcofs[1+m,1+i-j]*c1l*c2l
        end
        tmat[1+i,1+2*m+1] = adr
        tmat[1+i,1+m]=((-1.0)^(m+1))*adl
    end
    # Now loop over the other columns backwards
    
    for k=m-1:-1:0
        for i=0:2*m+1
            adl=0.0
            adr=0.0
            for j = max(0,i-k):min(i,m+1)
                if (k-i+j) == 0  
                    c2l = 1.0
                    c2r = 1.0
                else
                    c2l = z^(k-i+j)
                    c2r = zc^(k-i+j)
                end 
                if ((m+1-j) == 0)
                    c1l = 1.0
                    c1r = 1.0
                else
                    c1l = (zc^(m+1-j))
                    c1r = (z^(m+1-j))
                end 
                adr = adr + bcofs[1+m+1,1+j]*bcofs[1+k,1+i-j]*c1r*c2r
                adl = adl + bcofs[1+m+1,1+j]*bcofs[1+k,1+i-j]*c1l*c2l
            end
            tmat[1+i,1+k+m+1] = adr
            tmat[1+i,1+k] = ((-1.0)^(m+1))*adl
            sign = 1.0
            for j = k+1:m
                sign=-sign
                tmat[1+i,1+k]=tmat[1+i,1+k]-sign*bcofs[1+m+1,1+j-k]*tmat[1+i,1+j]
                tmat[1+i,1+k+m+1]=tmat[1+i,1+k+m+1]-bcofs[1+m+1,1+j-k]*tmat[1+i,1+j+m+1]
            end
        end
    end
end

function binomial!(coeffs,m)
    # Computes the binomial coefficients of order up through m
    coeffs[1,1] = 1.0
    for i = 1:m
        coeffs[1+i,1+0] = 1.0
        for j=1:i
            coeffs[1+i,1+j] = coeffs[1+i,1+j-1]*(i-j+1)/(j)
        end
    end
end

"""
Recenters a polynomial from 0 to z.
That is, rewrite `a0 + a1*x + a2*x^2 + ...` as `b0 + b1*(x-z) + b2*(x-z)^2 + ...`
"""
function extrapolate!(p,z,q,ploc)
    #=
    !
    ! recenters a polynomial from 0 to z.
    ! 
    !
    INTEGER, INTENT(IN) :: q
    DOUBLE PRECISION, DIMENSION(0:q), INTENT(INOUT) :: p
    DOUBLE PRECISION, INTENT(IN) :: z
    DOUBLE PRECISION, DIMENSION(0:q) :: ploc
    INTEGER :: j,k
    !
    =#
    ploc .= 0.0
    ploc[1+0] = p[1+q]
    for j=q-1:-1:0
        for k=q-j:-1:1
            ploc[1+k] = z*ploc[1+k]+ploc[1+k-1]
        end
        ploc[1+0] = z*ploc[1+0]+p[1+j]
    end
    # So the computation is all done to place the result in ploc, but then it is 
    # finally copied to p. I think it would be better to just keep it in ploc.
    p .= ploc
end

"""
Recenters a polynomial from 0 to z.
That is, rewrite `a0 + a1*x + a2*x^2 + ...` as `b0 + b1*(x-z) + b2*(x-z)^2 + ...`
"""
function extrapolate2!(poly_z, poly_0, z)
    #=
    !
    ! recenters a polynomial from 0 to z.
    ! 
    !
    INTEGER, INTENT(IN) :: q
    DOUBLE PRECISION, DIMENSION(0:q), INTENT(INOUT) :: p
    DOUBLE PRECISION, INTENT(IN) :: z
    DOUBLE PRECISION, DIMENSION(0:q) :: ploc
    INTEGER :: j,k
    !
    =#
    q = length(p)
    poly_z .= 0.0
    poly_z[1+0] = poly_0[1+q]
    for j=q-1:-1:0
        for k=q-j:-1:1
            poly_z[1+k] = z*poly_z[1+k]+poly_z[1+k-1]
        end
        poly_z[1+0] = z*poly_z[1+0]+poly_0[1+j]
    end
    return poly_z
end
