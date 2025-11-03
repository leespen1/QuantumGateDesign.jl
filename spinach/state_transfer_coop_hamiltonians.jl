using MATLAB, DelimitedFiles

# Add spinach paths
mat"""
addpath(genpath('~/Documents/MATLAB/Spinach/kernel'));
addpath(genpath('~/Documents/MATLAB/Spinach/etc'));
addpath(genpath('~/Documents/MATLAB/Spinach/experiments'));
addpath(genpath('~/Documents/MATLAB/Spinach/interfaces'));
"""

# Get the hamiltonians
H, Lx, Ly, rho_init, rho_targ = mxcall(:state_transfer_coop_hamiltonians, 5)
writedlm("H.dlm", H)
writedlm("Lx.dlm", Lx)
writedlm("Ly.dlm", Ly)
writedlm("rho_init.dlm", rho_init)
writedlm("rho_targ.dlm", rho_targ)
