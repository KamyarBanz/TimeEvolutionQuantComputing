from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

Boundary = Literal["open", "periodic"]


def pauli_dense() -> dict[str, np.ndarray]:
    i2 = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    return {"I": i2, "X": x, "Y": y, "Z": z, "H": h}


def rz_dense(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
        dtype=np.complex128,
    )


def equator_unitary_from_zero(phi: float) -> np.ndarray:
    p = pauli_dense()
    return rz_dense(phi) @ p["H"]


def basis_state(bitstring: str) -> np.ndarray:
    if any(ch not in "01" for ch in bitstring):
        raise ValueError("bitstring must contain only '0'/'1'")
    dim = 2 ** len(bitstring)
    index = int(bitstring, 2) if bitstring else 0
    state = np.zeros((dim,), dtype=np.complex128)
    state[index] = 1.0
    return state


def _as_sparse(op: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(op, dtype=np.complex128)


def kron_all_sparse(ops: Iterable[sp.spmatrix]) -> sp.csr_matrix:
    ops = list(ops)
    if not ops:
        raise ValueError("ops must be non-empty")
    out = ops[0].tocsr()
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


def two_site_term_sparse(
    op_i: sp.spmatrix, i: int, op_j: sp.spmatrix, j: int, L: int
) -> sp.csr_matrix:
    if not (0 <= i < L and 0 <= j < L):
        raise ValueError("site index out of range")
    if i == j:
        raise ValueError("i and j must be different sites")
    i2 = sp.identity(2, dtype=np.complex128, format="csr")
    ops: list[sp.spmatrix] = [i2] * L
    ops[i] = op_i
    ops[j] = op_j
    return kron_all_sparse(ops)


def xxz_hamiltonian_sparse(L: int, Jz: float, boundary: Boundary = "open") -> sp.csr_matrix:
    if L <= 0:
        raise ValueError("L must be positive")
    if boundary not in ("open", "periodic"):
        raise ValueError("boundary must be 'open' or 'periodic'")

    p = pauli_dense()
    x = _as_sparse(p["X"])
    y = _as_sparse(p["Y"])
    z = _as_sparse(p["Z"])

    if boundary == "open":
        pairs = [(i, i + 1) for i in range(L - 1)]
    else:
        pairs = [(i, (i + 1) % L) for i in range(L)]

    dim = 2**L
    H = sp.csr_matrix((dim, dim), dtype=np.complex128)

    for i, j in pairs:
        H = H - two_site_term_sparse(x, i, x, j, L)
        H = H - two_site_term_sparse(y, i, y, j, L)
        H = H - (Jz * two_site_term_sparse(z, i, z, j, L))

    return H


def apply_single_qubit_unitary(
    state: np.ndarray, U: np.ndarray, site: int, L: int
) -> np.ndarray:
    if state.shape != (2**L,):
        raise ValueError("state shape must be (2**L,)")
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")
    if not (0 <= site < L):
        raise ValueError("site index out of range")

    tensor = state.reshape((2,) * L)
    tensor = np.moveaxis(tensor, site, 0)  # (2, ...)
    tensor = np.tensordot(U, tensor, axes=([1], [0]))  # (2, ...)
    tensor = np.moveaxis(tensor, 0, site)
    return tensor.reshape((2**L,))


def local_expectation(state: np.ndarray, op: np.ndarray, site: int, L: int) -> complex:
    if state.shape != (2**L,):
        raise ValueError("state shape must be (2**L,)")
    if op.shape != (2, 2):
        raise ValueError("op must be a 2x2 matrix")
    if not (0 <= site < L):
        raise ValueError("site index out of range")

    ket = state.reshape((2,) * L)
    ket = np.moveaxis(ket, site, 0)
    ket = np.tensordot(op, ket, axes=([1], [0]))
    ket = np.moveaxis(ket, 0, site)
    return np.vdot(state, ket.reshape((2**L,)))


def evolve_states_expm_multiply(
    H: sp.spmatrix, state0: np.ndarray, times: np.ndarray
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")

    A = (-1j) * H

    if len(times) == 0:
        return np.zeros((0, state0.size), dtype=np.complex128)

    if len(times) == 1:
        return np.asarray([expm_multiply(A * times[0], state0)], dtype=np.complex128)

    dt = times[1] - times[0]
    grid = times[0] + dt * np.arange(len(times), dtype=float)
    if np.allclose(times, grid, rtol=0, atol=1e-12):
        states = expm_multiply(A, state0, start=times[0], stop=times[-1], num=len(times), endpoint=True)
        return np.asarray(states, dtype=np.complex128)

    out = np.empty((len(times), state0.size), dtype=np.complex128)
    for idx, t in enumerate(times):
        out[idx] = expm_multiply(A * t, state0)
    return out


@dataclass(frozen=True)
class SimulationConfig:
    L: int
    Jz: float
    boundary: Boundary = "open"
    phi: float = 0.0
    rotate_site: int | None = None


def initial_state_from_config(cfg: SimulationConfig) -> np.ndarray:
    if cfg.L <= 0:
        raise ValueError("L must be positive")
    # Default: all spins down |0>^{\otimes L}
    bitstring = "0" * cfg.L
    state = basis_state(bitstring)

    site = cfg.rotate_site if cfg.rotate_site is not None else (cfg.L // 2)
    U = equator_unitary_from_zero(cfg.phi)
    return apply_single_qubit_unitary(state, U, site=site, L=cfg.L)

