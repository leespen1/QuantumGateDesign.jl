
function PauliX_gate()
    return [0 1
            1 0]
end

function PauliY_gate()
    return [0 -im
            im 0]
end

function PauliZ_gate()
    return [1 0
            0 -1]
end

function Hadamard_gate()
    H = [1 1
         1 -1]
    return H ./ sqrt(2)
end

function Phase_gate()
    return [1 0
            0 im]
end

function T_gate()
    return [1 0
            0 exp(im*pi/4)]
end

function CNOT_gate()
    return [1 0 0 0
            0 1 0 0
            0 0 0 1
            0 0 1 0]
end

function SWAP_gate()
    return [1 0 0 0
            0 0 1 0
            0 1 0 0
            0 0 0 1]
end


function ControlledZ_gate()
    return [1 0 0 0
            0 1 0 0
            0 0 1 0
            0 0 0 -1]
end


"""
    QFT_gate(n_qubits)

Gate representation of the quantum Fourier transform, applied to `2^n_qubits`
amplitudes.
"""
function QFT_gate(n_qubits)
    N = 2^n_qubits
    gate = Matrix{ComplexF64}(undef, N, N)
    w = exp(2pi*im/N) # N-th root of unity

    for k=0:N-1
        for j=0:N-1
            gate[1+j,1+k] = w^(j*k)
        end
    end
    gate ./= sqrt(N)

    return gate
end

function Toffoli_gate()
    gate = zeros(Int64, 8, 8)
    for i in 1:6
        gate[i,i] = 1
    end
    gate[7,8] = 1
    gate[8,7] = 1
    return gate
end
