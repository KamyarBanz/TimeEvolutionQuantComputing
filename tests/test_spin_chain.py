"""Unit tests for GroupProject.time_evolution.spin_chain."""

from __future__ import annotations

import numpy as np
import pytest

from GroupProject.time_evolution.spin_chain import (
    SimulationConfig,
    apply_single_qubit_unitary,
    basis_state,
    evolve_states_expm_multiply,
    initial_state_from_config,
    kron_all_sparse,
    local_expectation,
    pauli_dense,
    rz_dense,
    two_site_term_sparse,
    xxz_hamiltonian_sparse,
)


# ---------------------------------------------------------------------------
# pauli_dense / rz_dense
# ---------------------------------------------------------------------------

class TestPauliDense:
    def test_keys(self):
        p = pauli_dense()
        assert set(p.keys()) == {"I", "X", "Y", "Z", "H"}

    def test_pauli_squares_to_identity(self):
        p = pauli_dense()
        for label in ("X", "Y", "Z"):
            np.testing.assert_allclose(p[label] @ p[label], p["I"], atol=1e-14)

    def test_hadamard_unitary(self):
        p = pauli_dense()
        np.testing.assert_allclose(p["H"] @ p["H"].conj().T, p["I"], atol=1e-14)


class TestRzDense:
    def test_rz_zero_is_identity(self):
        np.testing.assert_allclose(rz_dense(0.0), np.eye(2), atol=1e-14)

    def test_rz_is_unitary(self):
        phi = 1.23
        U = rz_dense(phi)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-14)


# ---------------------------------------------------------------------------
# basis_state
# ---------------------------------------------------------------------------

class TestBasisState:
    def test_single_qubit(self):
        s0 = basis_state("0")
        np.testing.assert_allclose(s0, [1, 0])

        s1 = basis_state("1")
        np.testing.assert_allclose(s1, [0, 1])

    def test_two_qubits(self):
        s = basis_state("10")
        expected = np.array([0, 0, 1, 0], dtype=np.complex128)
        np.testing.assert_allclose(s, expected)

    def test_invalid_bitstring(self):
        with pytest.raises(ValueError):
            basis_state("02")


# ---------------------------------------------------------------------------
# kron_all_sparse / two_site_term_sparse
# ---------------------------------------------------------------------------

class TestKronAllSparse:
    def test_single_op(self):
        import scipy.sparse as sp
        m = sp.eye(3, format="csr")
        result = kron_all_sparse([m])
        np.testing.assert_allclose(result.toarray(), np.eye(3))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            kron_all_sparse([])


class TestTwoSiteTermSparse:
    def test_shape(self):
        import scipy.sparse as sp
        p = pauli_dense()
        op = two_site_term_sparse(
            sp.csr_matrix(p["X"]), 0,
            sp.csr_matrix(p["Z"]), 1,
            L=3,
        )
        assert op.shape == (8, 8)

    def test_same_site_raises(self):
        import scipy.sparse as sp
        p = pauli_dense()
        with pytest.raises(ValueError):
            two_site_term_sparse(
                sp.csr_matrix(p["X"]), 1,
                sp.csr_matrix(p["Z"]), 1,
                L=3,
            )


# ---------------------------------------------------------------------------
# xxz_hamiltonian_sparse
# ---------------------------------------------------------------------------

class TestXXZHamiltonian:
    def test_hermitian(self):
        H = xxz_hamiltonian_sparse(L=4, Jz=1.0, boundary="open")
        diff = H - H.conj().T
        assert abs(diff).max() < 1e-14

    def test_periodic_hermitian(self):
        H = xxz_hamiltonian_sparse(L=4, Jz=0.5, boundary="periodic")
        diff = H - H.conj().T
        assert abs(diff).max() < 1e-14

    def test_invalid_L(self):
        with pytest.raises(ValueError):
            xxz_hamiltonian_sparse(L=0, Jz=1.0)


# ---------------------------------------------------------------------------
# apply_single_qubit_unitary / local_expectation
# ---------------------------------------------------------------------------

class TestApplySingleQubitUnitary:
    def test_identity_no_change(self):
        state = basis_state("01")
        result = apply_single_qubit_unitary(state, np.eye(2, dtype=complex), site=0, L=2)
        np.testing.assert_allclose(result, state, atol=1e-14)

    def test_x_gate_flips(self):
        p = pauli_dense()
        state = basis_state("00")
        result = apply_single_qubit_unitary(state, p["X"], site=0, L=2)
        np.testing.assert_allclose(result, basis_state("10"), atol=1e-14)


class TestLocalExpectation:
    def test_z_on_computational_basis(self):
        p = pauli_dense()
        state = basis_state("0")
        # |0> is +1 eigenstate of Z
        assert abs(local_expectation(state, p["Z"], site=0, L=1) - 1.0) < 1e-14

        state1 = basis_state("1")
        # |1> is -1 eigenstate of Z
        assert abs(local_expectation(state1, p["Z"], site=0, L=1) - (-1.0)) < 1e-14


# ---------------------------------------------------------------------------
# evolve_states_expm_multiply
# ---------------------------------------------------------------------------

class TestEvolveStates:
    def test_identity_evolution(self):
        import scipy.sparse as sp
        L = 2
        H = sp.csr_matrix((2**L, 2**L), dtype=np.complex128)
        state0 = basis_state("01")
        times = np.linspace(0, 1, 5)
        states = evolve_states_expm_multiply(H, state0, times)
        for s in states:
            np.testing.assert_allclose(s, state0, atol=1e-13)

    def test_empty_times(self):
        import scipy.sparse as sp
        H = sp.csr_matrix((4, 4), dtype=np.complex128)
        state0 = basis_state("00")
        states = evolve_states_expm_multiply(H, state0, np.array([]))
        assert states.shape == (0, 4)


# ---------------------------------------------------------------------------
# SimulationConfig / initial_state_from_config
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_initial_state_normalized(self):
        cfg = SimulationConfig(L=4, Jz=1.0, phi=0.5)
        state = initial_state_from_config(cfg)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-14

    def test_invalid_L(self):
        cfg = SimulationConfig(L=0, Jz=1.0)
        with pytest.raises(ValueError):
            initial_state_from_config(cfg)
