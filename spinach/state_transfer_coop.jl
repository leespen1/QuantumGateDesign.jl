using MATLAB

# Add spinach paths
mat"""
addpath(genpath('~/Documents/MATLAB/Spinach/kernel'));
addpath(genpath('~/Documents/MATLAB/Spinach/etc'));
addpath(genpath('~/Documents/MATLAB/Spinach/experiments'));
addpath(genpath('~/Documents/MATLAB/Spinach/interfaces'));
"""

# Get the hamiltonians
H, Lx, Ly = mxcall(:state_transfer_coop_hamiltonians, 3)
