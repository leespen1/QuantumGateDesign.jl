import Ipopt

# First set up constraint, jacobian, and hessian functions. We will not be
# using those in our optimization process (yet), so they all just return
# nothing.
#
# I have added exclamation points because their purpose really is to mutate
# vectors. They don't return anything meaningful.

function dummy_eval_g!(x::Vector{Float64}, g::Vector{Float64})
    return
end

function dummy_eval_jacobian_g!(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    values::Union{Nothing,Vector{Float64}},
)
    return
end

function dummy_eval_hessian!(
        x::Vector{Float64},
        rows::Vector{Int32},
        cols::Vector{Int32},
        obj_factor::Float64,
        lambda::Vector{Float64},
        values::Union{Nothing,Vector{Float64}},
    )

    return 
end

function optimize_gate(
        schro_prob::SchrodingerProb{M, VM}, target::VM, pcof_init::V;
    ) where {V<:AbstractVector{Float64}, VM<:AbstractVecOrMat{Float64}, M<:AbstractMatrix{Float64}}

    # Right now I am unnecessarily doing a full forward evolution to compute the
    # infidelity, when this should be done already in the gradient computation.
    function eval_f(pcof::Vector{Float64})
        history = eval_forward(schro_prob, pcof)
        QN = history[:,end,:]
        return infidelity(QN, target, schro_prob.N_ess_levels)
    end

    function eval_grad_f!(pcof::Vector{Float64}, grad_f::Vector{Float64})
        grad_f .= discrete_adjoint(schro_prob, target, pcof)
    end

    N_coeff = length(pcof_init)
    x_L = zeros(N_coeff)
    x_U = ones(N_coeff) .* 2

    N_constraints = 0
    g_L = Vector{Float64}()
    g_U = Vector{Float64}()

    nele_jacobian = 0
    nele_hessian = 0

    ipopt_prob = Ipopt.CreateIpoptProblem(
        N_coeff,
        x_L,
        x_U,
        N_constraints,
        g_L,
        g_U,
        nele_jacobian,
        nele_hessian,
        eval_f,
        dummy_eval_g!,
        eval_grad_f!,
        dummy_eval_jacobian_g!,
        dummy_eval_hessian!,
    )

    ipopt_prob.x .= pcof_init
    solvestat = Ipopt.IpoptSolve(ipopt_prob)
    # solvestat I think is stored in ipopt_prob as 'status'

    return ipopt_prob
end
