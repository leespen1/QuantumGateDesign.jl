FC = gfortran
FFLAGS = -c -O3 -shared -fPIC -fdefault-real-8

OBJ = bsplvb.o bsplvd.o
bspline_lib: $(OBJ)
	gfortran -shared -o bspline_lib.so $(OBJ)
	mv bspline_lib.so ../Controls/

bsplvb.o: bsplvb.f
	$(FC) $(FFLAGS) bsplvb.f

bsplvd.o: bsplvd.f
	$(FC) $(FFLAGS) bsplvd.f

clean:
	/bin/rm -f *.o
