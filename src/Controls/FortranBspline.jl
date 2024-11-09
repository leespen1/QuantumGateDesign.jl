struct FortranBSplineControl <: AbstractControl
    N_coeff::Int64
    tf::Float64
    max_order::Int64
    ilo::Int64
    knot_vector::Vector{Float64}
    aj::Vector{Float64}
    dl::Vector{Float64}
    dr::Vector{Float64}
    function FortranBSplineControl(knot_vector::AbstractVector{<: Real}, bcoef::AbstractVector{<: Real}, order::Integer)
        new()
    end
end



"""
https://www.netlib.org/pppack/
https://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=pppack%2Fbvalue.f

  from  * a practical guide to splines *  by c. de boor    
alls  interv

alculates value at  x  of  jderiv-th derivative of spline from b-repr.
  the spline is taken to be continuous from the right, EXCEPT at the
  rightmost knot, where it is taken to be continuous from the left.

******  i n p u t ******
  t, bcoef, n, k......forms the b-representation of the spline  f  to
        be evaluated. specifically,
  t.....knot sequence, of length  n+k, assumed nondecreasing.
  bcoef.....b-coefficient sequence, of length  n .
  n.....length of  bcoef  and dimension of spline(k,t),
        a s s u m e d  positive .
  k.....order of the spline .

  w a r n i n g . . .   the restriction  k .le. kmax (=20)  is imposed
        arbitrarily by the dimension statement for  aj, dl, dr  below,
        but is  n o w h e r e  c h e c k e d  for.

  x.....the point at which to evaluate .
  jderiv.....integer giving the order of the derivative to be evaluated
        a s s u m e d  to be zero or positive.

******  o u t p u t  ******
  bvalue.....the value of the (jderiv)-th derivative of  f  at  x .

******  m e t h o d  ******
     The nontrivial knot interval  (t(i),t(i+1))  containing  x  is lo-
  cated with the aid of  interv . The  k  b-coeffs of  f  relevant for
  this interval are then obtained from  bcoef (or taken to be zero if
  not explicitly available) and are then differenced  jderiv  times to
  obtain the b-coeffs of  (d**jderiv)f  relevant for that interval.
  Precisely, with  j = jderiv, we have from x.(12) of the text that

     (d**j)f  =  sum ( bcoef(.,j)*b(.,k-j,t) )

  where
                   / bcoef(.),                     ,  j .eq. 0
                   /
    bcoef(.,j)  =  / bcoef(.,j-1) - bcoef(.-1,j-1)
                   / ----------------------------- ,  j .gt. 0
                   /    (t(.+k-j) - t(.))/(k-j)

     Then, we use repeatedly the fact that

    sum ( a(.)*b(.,m,t)(x) )  =  sum ( a(.,x)*b(.,m-1,t)(x) )
  with
                 (x - t(.))*a(.) + (t(.+m-1) - x)*a(.-1)
    a(.,x)  =    ---------------------------------------
                 (x - t(.))      + (t(.+m-1) - x)

  to write  (d**j)f(x)  eventually as a linear combination of b-splines
  of order  1 , and the coefficient for  b(i,1,t)(x)  must then be the
  desired number  (d**j)f(x). (see x.(17)-(19) of text).

"""
function bvalue(t::Vector{Float64}, bcoef::Vector{Float64}, n::Integer, 
        k::Integer, x::Real, jderiv::Integer, aj=zeros(kmax), dl=zeros(kmax), dr=zeros(kmax))::Real
      #integer jderiv,k,n,i,ilo,imk,j,jc,jcmin,jcmax,jj,kmax,kmj,km1,mflag,nmi,jdrvp1
      kmax = 20
      @assert length(bcoef) == n
      @assert length(t) == n+k


      bvalue = 0.
      if (jderiv >= k)#                go to 99
          return bvalue
      end
#
#  *** Find  i   s.t.   1 .le. i .lt. n+k   and   t(i) .lt. t(i+1)   and
#      t(i) .le. x .lt. t(i+1) . If no such i can be found,  x  lies
#      outside the support of  the spline  f , hence  bvalue = 0.
#      (The asymmetry in this choice of  i  makes  f  rightcontinuous, except
#      at  t(n+k) where it is leftcontinuous.)
#  *** Find i s.t. 1 <= i <= n+1 and t[i] < t[i+1] and t[i] <= x <= t[i+1].
#      If no such i can be found, then x lies outside the support of the spline
#      f, hence bvalue = 0. 
#      (The asymmetry in this choice of  i  makes  f  rightcontinuous, except
#      at  t[n+k] where it is leftcontinuous.)
      i, mflag = interv(t, n+k, x)
      if (mflag != 0)#                 go to 99
          return bvalue
      end

#  *** if k = 1 (and jderiv = 0), bvalue = bcoef(i).
      km1 = k - 1
      if (km1 <= 0)#                   go to 1
          bvalue = bcoef[i]
          return bvalue
      end
#
#  *** store the k b-spline coefficients relevant for the knot interval
#     (t(i),t(i+1)) in aj(1),...,aj(k) and compute dl(j) = x - t(i+1-j),
#     dr(j) = t(i+j) - x, j=1,...,k-1 . set any of the aj not obtainable
#     from input to zero. set any t.s not obtainable equal to t(1) or
#     to t(n+k) appropriately.
      jcmin = 1 #1
      imk = i - k
      if (imk >= 0)#                   go to 8
          for j=1:km1 #8
             dl[j] = x - t[i+1-j] #9
          end
      else
          jcmin = 1 - imk
          for j=1:i
             dl[j] = x - t[i+1-j] #5
          end
          for j=i:km1 # Says do 6, but the following line should still execute, right? Maybe do 6 means do until line 6. It deliminates where the do-loop ends
             aj[k-j] = 0.
             dl[j] = dl[i] #6
          end
      end
#
      jcmax = k #10
      nmi = n - i
      if (nmi >= 0)#                   go to 18
         for j=1:km1 #18
             dr[j] = t[i+j] - x #19
         end
      else
          jcmax = k + nmi
          for j=1:jcmax 
             dr[j] = t[i+j] - x #15
          end
          for j=jcmax:km1
             aj[j+1] = 0.
             dr[j] = dr[jcmax] #16
          end
      end
#
      for jc=jcmin:jcmax #20
         aj[jc] = bcoef[imk + jc] #21
      end
      
#
#               *** difference the coefficients  jderiv  times.
      if (jderiv != 0)#                go to 30 (but statement was originally ==)
          for j=1:jderiv
             kmj = k-j
             fkmj = Float64(kmj)
             ilo = kmj
             for jj=1:kmj
                aj[jj] = ((aj[jj+1] - aj[jj])/(dl[ilo] + dr[jj]))*fkmj
                ilo = ilo - 1 #23
             end
          end
      end
#
#  *** compute value at  x  in (t(i),t(i+1)) of jderiv-th derivative,
#     given its relevant b-spline coeffs in aj(1),...,aj(k-jderiv).
      if (jderiv == km1)#30              go to 39
          bvalue = aj[1]
          return bvalue
      end

      jdrvp1 = jderiv + 1     
      for j=jdrvp1:km1
         kmj = k-j
         ilo = kmj
         for jj=1:kmj
            aj[jj] = (aj[jj+1]*dl[ilo] + aj[jj]*dr[jj])/(dl[ilo]+dr[jj])
            ilo = ilo - 1 #33
         end
      end
      bvalue = aj[1]#39
#
      return bvalue #99
end

"""
  from  * a practical guide to splines *  by C. de Boor    
omputes  left = max( i :  xt(i) .lt. xt(lxt) .and.  xt(i) .le. x )  .

******  i n p u t  ******
  xt.....a real sequence, of length  lxt , assumed to be nondecreasing
  lxt.....number of terms in the sequence  xt .
  x.....the point whose location with respect to the sequence  xt  is
        to be determined.

******  o u t p u t  ******
  left, mflag.....both integers, whose value is

   1     -1      if               x .lt.  xt(1)
   i      0      if   xt(i)  .le. x .lt. xt(i+1)
   i      0      if   xt(i)  .lt. x .eq. xt(i+1) .eq. xt(lxt)
   i      1      if   xt(i)  .lt.        xt(i+1) .eq. xt(lxt) .lt. x

        In particular,  mflag = 0  is the 'usual' case.  mflag .ne. 0
        indicates that  x  lies outside the CLOSED interval
        xt(1) .le. y .le. xt(lxt) . The asymmetric treatment of the
        intervals is due to the decision to make all pp functions cont-
        inuous from the right, but, by returning  mflag = 0  even if
        x = xt(lxt), there is the option of having the computed pp function
        continuous from the left at  xt(lxt) .

******  m e t h o d  ******
  The program is designed to be efficient in the common situation that
  it is called repeatedly, with  x  taken from an increasing or decrea-
  sing sequence. This will happen, e.g., when a pp function is to be
  graphed. The first guess for  left  is therefore taken to be the val-
  ue returned at the previous call and stored in the  l o c a l  varia-
  ble  ilo . A first check ascertains that  ilo .lt. lxt (this is nec-
  essary since the present call may have nothing to do with the previ-
  ous call). Then, if  xt(ilo) .le. x .lt. xt(ilo+1), we set  left =
  ilo  and are done after just three comparisons.
     Otherwise, we repeatedly double the difference  istep = ihi - ilo
  while also moving  ilo  and  ihi  in the direction of  x , until
                      xt(ilo) .le. x .lt. xt(ihi) ,
  after which we use bisection to get, in addition, ilo+1 = ihi .
  left = ilo  is then returned.
"""
function interv(xt::Vector{Float64}, lxt::Integer, x::Real, ilo::Integer=1)
      @assert length(xt) == lxt
      #integer left,lxt,mflag,   ihi,ilo,istep,middle
      #real x,xt(lxt)
      
      ##### Not sure how this works. is ilo local to the function?
      #data ilo /1/
      #save ilo  

    ihi = ilo + 1
    if (ihi < lxt)#                 go to 20
        @goto l20
    end

    if (x >= xt[lxt])#            go to 110
        @goto l110
    end

    if (lxt <= 1)#                go to 90
        @goto l90
    end

    ilo = lxt - 1
    ihi = lxt


    @label l20
    if (x >= xt[ihi])#              go to 40
        @goto l40
    end

    if (x >= xt[ilo])#               go to 100
        @goto l100
    end

      #         **** now x .lt. xt(ilo) . decrease  ilo  to capture  x .
    istep = 1
    @label l31
    ihi = ilo
    ilo = ihi - istep
    if (ilo <= 1)#                go to 35
        @goto l35
    end

    if (x >= xt[ilo])#            go to 50
        @goto l50
    end
    istep = istep*2
                                        #go to 31
    @goto l31

    @label l35
    ilo = 1
    if (x < xt[1])#                 go to 90
        @goto l90
    end
    #                                    go to 50
    @goto l50
      #         **** now x .ge. xt(ihi) . increase  ihi  to capture  x .
    @label l40
    istep = 1
    @label l41
    ilo = ihi
    ihi = ilo + istep
    if (ihi >= lxt)#              go to 45
        @goto l45
    end

    if (x < xt[ihi])#            go to 50
        #goto l50
    end
    istep = istep*2
    #                                    go to 41
    @goto l41
    
    @label l45
    if (x >= xt[lxt])#               go to 110
        @goto l100
    end
    ihi = lxt
 
    #      **** now xt(ilo) .le. x .lt. xt(ihi) . narrow the interval.
    @label l50
    middle = (ilo + ihi)/2

    if (middle == ilo)#              go to 100
        @goto l100
    end
      #note. it is assumed that middle = ilo in case ihi = ilo+1 .
    if (x < xt[middle])#            go to 53
        @goto l53
    end
    ilo = middle
    #                                    go to 50
    @goto l50

    @label l53
    ihi = middle
    #                                    go to 50
    @goto l50

#**** set output and return.
    @label l90
    mflag = -1
    left = 1
    return left, mflag

    @label l100
    mflag = 0
    left = ilo
    return left, mflag

    @label l110
    mflag = 1
    if (x == xt[lxt])
        mflag = 0
    end
    left = lxt

    @label l111
    if (left == 1)
        return left, mflag
    end

	left = left - 1
	if (xt(left) < xt[lxt])
        return left, mflag
    end
										#go to 111
    @goto l111
end

