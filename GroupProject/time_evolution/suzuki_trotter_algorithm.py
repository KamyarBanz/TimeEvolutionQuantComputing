import numpy as np


# --- Pauli matrices ---
pauli_x = np.array([[0,1],[1,0]], dtype=complex)
pauli_y = np.array([[0,-1j],[1j,0]], dtype=complex)
pauli_z = np.array([[1,0],[0,-1]], dtype=complex)


def h_bond(J: float) -> np.ndarray:
    """
    Construct the Hamiltonian for each bond in a 1D spin/qubit chain with periodic boundary conditions.

    Parameters:
    J (float): Coupling strength between neighboring spins.
    L (int): Length of the spin chain.

    Returns:
    ndarray: Hamiltonian of the local system.

    """

    return -(np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y) + J*np.kron(pauli_z, pauli_z))


def bond_layers(L: int):
    """
    Returns two lists of nearest-neighbour bonds (i,j):
      even layer: (0,1),(2,3),...
      odd  layer: (1,2),(3,4),...
    """
    if L < 2:
        raise ValueError("L must be >= 2")
    if L % 2 != 0:
        raise ValueError("Periodic 2-layer bond split requires even L")
    
    even = [(i, i+1) for i in range(0, L-1, 2)]
    odd  = [(i, i+1) for i in range(1, L-1, 2)]

    # wrap bond (L-1,0) exists
    if (L - 1) % 2 == 0:
        even.append((L-1, 0))
    else:
        odd.append((L-1, 0))

    return even, odd


def U_bond(h: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute the unitary operator for a given bond Hamiltonian and time step.

    Parameters:
    h_bond (ndarray): Hamiltonian of the bond.
    dt (float): Time step for evolution.

    Returns:
    ndarray: Unitary operator for the bond.
    """
    w, V = np.linalg.eigh(h)  # h_bond is Hermitian
    return V @ np.diag(np.exp(-1j * w * dt)) @ V.conj().T


def trotter_schedule(L: int):
    even, odd = bond_layers(L)
    
    # describes the order + timestep factors for 2nd order
    return [
        (even, 0.5),
        (odd,  1.0),
        (even, 0.5),
    ]


def apply_two_qubit_unitary(psi: np.ndarray, U: np.ndarray, i: int, L: int) -> np.ndarray:
    """
    Apply a 4x4 unitary U to qubits (i, j) of an L-qubit statevector psi (size 2^L).
    Works for any i, j (including non-adjacent), but O(2^L).
    """
    j = (i + 1) % L  # periodic boundary conditions

    # Converts from statevector to tensor with L indices of dimension 2
    psi_t = psi.reshape([2] * L)

    # Move axes i and j to the front
    axes = (i, j) + tuple(k for k in range(L) if k not in (i, j))
    psi_perm = np.transpose(psi_t, axes).reshape(4, -1)
    psi_perm = U @ psi_perm

    # Undo permutation
    psi_perm = psi_perm.reshape([2, 2] + [2] * (L - 2))
    inv_axes = np.argsort(axes)
    psi_out = np.transpose(psi_perm, inv_axes).reshape(-1)

    return psi_out


def trotter_step(psi: np.ndarray, L: int, J: float, dt: float) -> np.ndarray:
    """One 2nd-order Suzuki–Trotter step for H = sum_bonds h_bond."""
    h = h_bond(J)

    for bonds, factor in trotter_schedule(L):
        U = U_bond(h, dt * factor)
        for (i, _) in bonds:
            psi = apply_two_qubit_unitary(psi, U, i, L)

    return psi


if __name__ == "__main__":
    L = 4      # must be even with your current bond_layers
    J = 1.0
    dt = 0.01
    steps = 100

    # setting initial state 
    psi = np.zeros(2**L, dtype=complex)
    psi[int("0000", 2)] = 1.0
    psi[int("1111", 2)] = 1.0
    psi[int("1010", 2)] = 1.0
    psi /= np.linalg.norm(psi)

    for _ in range(steps):
        psi = trotter_step(psi, L, J, dt)

    print("Final norm:", np.linalg.norm(psi))
    print("Amplitudes:", psi[:])