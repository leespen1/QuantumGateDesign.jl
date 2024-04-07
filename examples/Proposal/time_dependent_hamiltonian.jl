using QuantumGateDesign

system_sym = zeros(2,2)
system_asym = zeros(2,2)
a = [0.0 1;0 1]
sym_op = a + a'
asym_op = a - a'
sym_ops = [sym_op]
asym_ops = [asym_op]
u0 = [1.0 0; 0 1]
v0 = zeros(2,2)
tf = 1

tf = 1.0
nsteps = 100
N_ess_levels = 2

prob = SchrodingerProb(
    system_sym,
    system_asym,
    sym_ops,
    asym_ops,
    u0,
    v0,
    tf,
    nsteps,
    N_ess_levels
)

control = QuantumGateDesign.SinCosControl(tf, frequency=pi*tf)

V_tg = [0.0 1
        0 1
        0 0
        0 0]
