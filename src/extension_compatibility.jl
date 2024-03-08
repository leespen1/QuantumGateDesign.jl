#===============================================================================
#
# Defining functions without methods so that extensions can override them.
# May be a better way to do this.
#
===============================================================================#

# ControlVisualizer.jl

function visualize_control end



# DifferentialEquationsInterface.jl

function construct_ODEProb end



# QuTipIntegration.jl

function convert_to_numpy end
function Qobj end
function unpack_Qobj end
function simulate_prob_no_control end
