using GLMakie
using QuantumGateDesign

prob = QuantumGateDesign.rotating_frame_qubit(2, 0, tf=10.0, nsteps=1000,
                                              detuning_frequency=0.0,
                                              self_kerr_coefficient=0.0)

control = QuantumGateDesign.GRAPEControl(1, prob.tf)
controls = [control, control]

target = [0.0 1; 1 0; 0 0; 0 0]

visualize_control(controls, prob=prob, target=target)
