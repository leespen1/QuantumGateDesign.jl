% Gets the hamiltonians for a quadrupolar 14N spin at a fixed orientation
% and power level as done in `state_transfer_coop.m`, but here we do not 
% do any optimal control, we only construct the Hamiltonians.
% Write the results to files.
function [H, Lx, Ly, rho_init, rho_targ] = state_transfer_coop_hamiltonians()

% Magnet field
sys.magnet=14.1; 

% Isotopes
sys.isotopes={'14N'}; 

% Glycine NQI, random orientation
euler_angles=[1.0 2.0 3.0];
inter.coupling.matrix{1,1}=eeqq2nqi(1.18e6,0.53,1,euler_angles);

% Glycine 14N chemical shift
inter.zeeman.scalar{1}=32.4;

% Basis set
bas.formalism='sphten-liouv';
bas.approximation='none'; 

% Run Spinach housekeeping
spin_system=create(sys,inter);
spin_system=basis(spin_system,bas);

% Set up and normalise the initial state
rho_init=state(spin_system,'T1,0','14N');
rho_init=rho_init/norm(full(rho_init),2);

% Set up and normalise the target state
rho_targ=state(spin_system,'T2,0','14N');
rho_targ=rho_targ/norm(full(rho_targ),2);

% spin_system assumptions
spin_system=assume(spin_system,'qnmr');

% Get the drift Hamiltonian
[Iso,Q]=hamiltonian(spin_system);
H=Iso+orientation(Q,[1 2 3]);
C=carrier(spin_system,'14N');
H=rotframe(spin_system,C,H,'14N',2);

% Get the control operators
Lx=operator(spin_system,'Lx','14N');
Ly=operator(spin_system,'Ly','14N');

end
