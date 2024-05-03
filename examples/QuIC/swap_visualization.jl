using GLMakie
using QuantumGateDesign

#prob = QuantumGateDesign.rotating_frame_qubit(2, 0, tf=10.0, nsteps=1000,
#                                              detuning_frequency=0.0,
#                                              self_kerr_coefficient=0.0)
prob = QuantumGateDesign.construct_rabi_prob(tf=pi)

control = QuantumGateDesign.GRAPEControl(1, prob.tf)

target = [0.0 1; 1 0; 0 0; 0 0]

visualize_control(control, prob=prob, target=target)
