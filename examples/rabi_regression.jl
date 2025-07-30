using QuantumGateDesign

H_drift = zeros(2,2)
H_control_real = [0.0 1;
                  1   0]
H_control_imag = [0.0 1;
                  -1  0]
real_control_ops = [H_control_real]
imag_control_ops = [H_control_imag]

#psi0 = reshape([1.0, 0], :, 1)
psi0 = [1.0, 0]
T = 50.0 # time in nanoseconds
nsteps = 100
sym_ops = [H_control_real]
asym_ops = [H_control_imag]
prob = SchrodingerProb(H_drift, real_control_ops, imag_control_ops, psi0, T, nsteps)

N_GRAPE_amplitudes = 1
control = GRAPEControl(N_GRAPE_amplitudes, T)
pcof = [pi/(2*T), 0]

history = eval_forward(prob, control, pcof)
pcof_init = zeros(2)
target = [0.0, 1]

pl = plot_controls(control, pcof)
