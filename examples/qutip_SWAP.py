import numpy as np
import matplotlib.pyplot as plt
import datetime



from qutip import Qobj, destroy
import qutip.logging_utils as logging
logger = logging.get_logger()
#Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#QuTiP control modules
import qutip.control.pulseoptim as cpo



example_name = 'SWAP'



## Hamiltonian Setup
# Set up "drift"/"control" Hamiltonian, non-tunable
d = 3
N_ess_levels = d+1
N_guard_levels = 1
N_tot_levels = N_ess_levels + N_guard_levels

# Frequencies in GHz
detuning_frequency = 0
self_kerr_coefficient = 0.22

# Maximum control amplitude
amp_ubound = 9*(1e-0)*2*np.pi # Set amplitude bounds of 9 MHz, multiply by 2pi for angular frequency
amp_lbound = -amp_ubound

a = destroy(N_tot_levels)

# Construct system or "drift" Hamiltonian (frequencies should be angular, so multiply by 2pi)
H_d = 2*np.pi*detuning_frequency * (a.dag()*a) - 0.5*2*np.pi*self_kerr_coefficient*(a.dag()*a.dag()*a*a)

# Set up tunable/control Hamiltonians
H_c = [a + a.dag(), a - a.dag()]

## Set up initial basis and target gate 
#U_0_np = np.zeros((N_tot_levels, N_tot_levels))
#for i in range(N_tot_levels):
U_0_np = np.zeros((N_tot_levels, N_ess_levels))
for i in range(N_ess_levels):
    U_0_np[i,i] = 1

U_0 = Qobj(U_0_np)

U_targ_np = np.copy(U_0_np)
U_targ_np[0,0] = 0
U_targ_np[N_ess_levels-1,0] = 1
U_targ_np[N_ess_levels-1,N_ess_levels-1] = 0
U_targ_np[0,N_ess_levels-1] = 1

U_targ = Qobj(U_targ_np)


evo_time = 140 # ns
n_timesteps = 4480 # = 0.5*n_parameters



## Optimization paramters
# Fidelity error target
fid_err_targ = 1e-6
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 60*1
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-20



# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
# This is what we use as the original guess for the pulse. It's always
# piecewise constant, but we can chose the initial amplitudes to be random, 
# zero, a piecewise constant approximation of a sine wave, etc

# I think zero is a good choice, since I can have the same start for GRAPE and Hermite
#p_type = 'RND'
p_type = 'RND'



#Set to None to suppress output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_timesteps, p_type)


# cpo.optimize_pulse_unitary exists, but I'm not sure if I should be using that
result = cpo.optimize_pulse(H_d, H_c, U_0, U_targ, n_timesteps, evo_time, 
                fid_err_targ=fid_err_targ, min_grad=min_grad, 
                max_iter=max_iter, max_wall_time=max_wall_time, 
                out_file_ext=f_ext, init_pulse_type=p_type, 
                log_level=log_level, gen_stats=True,
                amp_ubound=amp_ubound, amp_lbound=amp_lbound)



result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
        datetime.timedelta(seconds=result.wall_time)))

# Convert Amplitudes to MHz and non-angular for plotting
result.initial_amps *= (1e3/(2*np.pi))
result.final_amps *= (1e3/(2*np.pi))

fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial Control Amplitudes")
#ax1.set_xlabel("Time")
ax1.set_ylabel("Control Amplitude (MHz)")
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 1], result.initial_amps[-1, 1])),
         where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Control Amplitude (MHz)")
ax2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
ax2.step(result.time,
         np.hstack((result.final_amps[:, 1], result.final_amps[-1, 1])),
         where='post')
plt.tight_layout()
plt.show()
