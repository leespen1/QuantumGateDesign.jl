import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmay, sigmaz, tensor
from qutip.qip.algorithms import qft
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

example_name = 'QFT'

# Defining the Physics

Sx = sigmax()
Sy = sigmay()
Sz = sigmaz()
Si = 0.5*identity(2)

# Drift Hamiltonian
H_d = 0.5*(tensor(Sx, Sx) + tensor(Sy, Sy) + tensor(Sz, Sz))
# The (four) control Hamiltonians
H_c = [tensor(Sx, Si), tensor(Sy, Si), tensor(Si, Sx), tensor(Si, Sy)]
n_ctrls = len(H_c)
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution - Quantum Fourier Transform gate
U_targ = qft.qft(2)
print("Drift Hamiltonian:")
print(H_d)
print("Control Hamiltonians:")
for mat in H_c:
    print(H_c)
print("U_0:")
print(U_0)
print("U_targ:")
print(U_targ)

# Defining the Time Evolution Parameters

# Duration of each timeslot
dt = 0.05
# List of evolution times to try
evo_time = 10
num_tslots = int(float(evo_time) / dt)
print(f"dt: {dt}")
print(f"evo_time: {evo_time}")
print(f"numt_tslots: {num_tslots}")

# Set the conditions which will cause the pulse optimization to terminate

# Fidelity error target
fid_err_targ = 1e-5
# Maximum iterations for the optisation algorithm
max_iter = 200
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20
print(f"fid_err_targ: {fid_err_targ}")
print(f"max_iter: {max_iter}")
print(f"max_wall_time: {max_wall_time}")
print(f"min_grad: {min_grad}")

# Set the initial pulse type

# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'

ret = cpo.optimize_pulse(H_d, H_c, U_0, U_targ, num_tslots=num_tslots, evo_time=evo_time)
