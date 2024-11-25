using QuantumGateDesign.jl, ArgParse, Dates, Random

# Need to add something setting up the prob, controls, target, here
# E.g. Run setup
include("cnot3_setup.jl")

function main(prob::SchrodingerProb)

s = ArgParseSettings()

@add_arg_table s begin
    "--use_juqbox", "-j"
        help = "Flag for using Juqbox"
        action = :store_true
    "--gmres_abstol", "-a"
        help = "Absolute tolerance to use in GMRES"
        arg_type = Float64
        default = 1e-12
    "--gmres_reltol", "-r"
        help = "Relative tolerance to use in GMRES"
        arg_type = Float64
        default = 1e-12
    "--max_iter", "-m"
        help = "Maximum number of IPOPT iterations."
        arg_type = Int
        default = 50
    "--seed", "-s"
        help = "Random seed to use in control vector initialization"
        arg_type = Int
        default = 0
    "order"
        help = "Order of the method."
        arg_type = Int
        required = true
    "stepsize" # Change this to points per wavelength
        help = "Stepsize to use."
        arg_type = Float64
        required = true
end

parsed_args = parse_args(s)


order = parsed_args["order"]
stepsize = parsed_args["stepsize"]
use_juqbox = parsed_args["use_juqbox"]
max_iter = parsed_args["max_iter"]
gmres_abstol = parsed_args["gmres_abstol"]
gmres_reltol = parsed_args["gmres_reltol"]
seed = parsed_args["seed"]


N_coeffs = get_number_of_control_parameters(controls)
# Generate a random initial control vector
pcof_init = rand(MersenneTwister(seed), N_coeffs)
# Shift the initial control vector to be within amplitude bounds
pcof_init .-= 0.5
pcof_init .*= max_amplitude


# Compute nsteps based on command line input 
# Should change to points per wavelength, and have it warn if it changes
# drastically due to rounding.
nsteps = ceil(Int, prob.tf / stepsize)

prob.nsteps = nsteps


end
