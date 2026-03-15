from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

Boundary = Literal["open", "periodic"]
InitPattern = Literal["all0", "all1", "alternating10"]

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector


def _import_qiskit() -> tuple[type["QuantumCircuit"], type["Statevector"], type["SparsePauliOp"]]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except ImportError as exc:  # pragma: no cover - only triggered when qiskit missing
        raise ImportError(
            "qiskit is required for GroupProject.time_evolution.qc. "
            "Install it with: pip install qiskit"
        ) from exc
    return QuantumCircuit, Statevector, SparsePauliOp


def _site_to_qubit(site: int, L: int) -> int:
    if not (0 <= site < L):
        raise ValueError("site index out of range")
    # Keep site ordering consistent with spin_chain.py (site 0 is left-most/MSB).
    return L - 1 - site


def build_initial_bitstring(L: int, init_pattern: InitPattern) -> str:
    if L <= 0:
        raise ValueError("L must be positive")
    if init_pattern == "all0":
        return "0" * L
    if init_pattern == "all1":
        return "1" * L
    if init_pattern == "alternating10":
        if L % 2 != 0:
            raise ValueError("alternating10 requires even L")
        return "10" * (L // 2)
    raise ValueError(f"unknown init_pattern: {init_pattern}")


def build_bonds(L: int, boundary: Boundary) -> list[tuple[int, int]]:
    if L <= 1:
        raise ValueError("L must be at least 2")
    if boundary == "open":
        raw = [(i, i + 1) for i in range(L - 1)]
    elif boundary == "periodic":
        raw = [(i, (i + 1) % L) for i in range(L)]
    else:
        raise ValueError("boundary must be 'open' or 'periodic'")

    bonds: list[tuple[int, int]] = []
    for i, j in raw:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in bonds:
            bonds.append((a, b))
    return bonds


@dataclass(frozen=True)
class QiskitSimulationConfig:
    L: int
    Jz: float
    boundary: Boundary = "open"
    init_pattern: InitPattern = "all0"
    phi: float = 0.0
    rotate_site: int | None = None


def build_initial_circuit(
    L: int,
    init_pattern: InitPattern,
    phi: float,
    rotate_site: int | None = None,
) -> tuple["QuantumCircuit", str, int]:
    QuantumCircuit, _, _ = _import_qiskit()

    bitstring = build_initial_bitstring(L=L, init_pattern=init_pattern)
    site = rotate_site if rotate_site is not None else (L // 2)
    if not (0 <= site < L):
        raise ValueError("rotate_site out of range")

    qc = QuantumCircuit(L)

    for site_idx, bit in enumerate(bitstring):
        if bit == "1":
            qc.x(_site_to_qubit(site_idx, L))

    q = _site_to_qubit(site, L)
    if bitstring[site] == "0":
        qc.h(q)
        qc.rz(phi, q)
    else:
        qc.x(q)
        qc.h(q)
        qc.rz(phi, q)

    return qc, bitstring, site


def initial_statevector(
    L: int,
    init_pattern: InitPattern,
    phi: float,
    rotate_site: int | None = None,
) -> tuple[np.ndarray, str, int]:
    _, Statevector, _ = _import_qiskit()
    init_circuit, bitstring, site = build_initial_circuit(
        L=L, init_pattern=init_pattern, phi=phi, rotate_site=rotate_site
    )
    state = Statevector.from_instruction(init_circuit)
    return np.asarray(state.data, dtype=np.complex128), bitstring, site


def append_xxz_two_site_evolution(
    circuit: "QuantumCircuit",
    i: int,
    j: int,
    dt: float,
    Jz: float,
    L: int,
) -> None:
    if i == j:
        raise ValueError("two-site term requires i != j")
    qi = _site_to_qubit(i, L)
    qj = _site_to_qubit(j, L)

    # qiskit definitions:
    # rxx(theta) = exp(-i * theta/2 * (X kron X)), same for ryy/rzz.
    # We need exp(+i * dt * (XX + YY + Jz ZZ)).
    circuit.rxx(-2.0 * dt, qi, qj)
    circuit.ryy(-2.0 * dt, qi, qj)
    circuit.rzz(-2.0 * Jz * dt, qi, qj)


def append_trotter_interval(
    circuit: "QuantumCircuit",
    L: int,
    Jz: float,
    dt: float,
    boundary: Boundary = "open",
    order: int = 2,
) -> None:
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if L <= 1:
        raise ValueError("L must be at least 2")

    bonds = build_bonds(L=L, boundary=boundary)
    if order == 1:
        for i, j in bonds:
            append_xxz_two_site_evolution(circuit, i=i, j=j, dt=dt, Jz=Jz, L=L)
        return

    half = 0.5 * dt
    for i, j in bonds:
        append_xxz_two_site_evolution(circuit, i=i, j=j, dt=half, Jz=Jz, L=L)
    for i, j in reversed(bonds):
        append_xxz_two_site_evolution(circuit, i=i, j=j, dt=half, Jz=Jz, L=L)


def build_time_evolution_circuit(
    cfg: QiskitSimulationConfig,
    t_final: float,
    n_steps: int,
    order: int = 2,
) -> tuple["QuantumCircuit", str, int]:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    init_circuit, bitstring, site = build_initial_circuit(
        L=cfg.L,
        init_pattern=cfg.init_pattern,
        phi=cfg.phi,
        rotate_site=cfg.rotate_site,
    )

    if n_steps == 0:
        return init_circuit, bitstring, site

    dt = float(t_final) / float(n_steps)
    for _ in range(n_steps):
        append_trotter_interval(
            init_circuit,
            L=cfg.L,
            Jz=cfg.Jz,
            dt=dt,
            boundary=cfg.boundary,
            order=order,
        )

    return init_circuit, bitstring, site


def evolve_trotter_states_qiskit(
    state0: np.ndarray,
    L: int,
    Jz: float,
    boundary: Boundary,
    times: np.ndarray,
    order: int = 2,
) -> np.ndarray:
    QuantumCircuit, Statevector, _ = _import_qiskit()

    times = np.asarray(times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    if state0.shape != (2**L,):
        raise ValueError("state0 shape must be (2**L,)")
    if len(times) == 0:
        return np.zeros((0, state0.size), dtype=np.complex128)

    states = np.empty((len(times), state0.size), dtype=np.complex128)
    current = Statevector(state0)
    states[0] = np.asarray(current.data, dtype=np.complex128)

    step_cache: dict[float, QuantumCircuit] = {}
    for idx in range(1, len(times)):
        dt = float(times[idx] - times[idx - 1])
        cache_key = float(np.round(dt, 15))
        if cache_key not in step_cache:
            step = QuantumCircuit(L)
            append_trotter_interval(
                step, L=L, Jz=Jz, dt=dt, boundary=boundary, order=order
            )
            step_cache[cache_key] = step
        current = current.evolve(step_cache[cache_key])
        states[idx] = np.asarray(current.data, dtype=np.complex128)

    return states


def build_local_pauli_ops(L: int) -> tuple[list["SparsePauliOp"], list["SparsePauliOp"], list["SparsePauliOp"]]:
    if L <= 0:
        raise ValueError("L must be positive")
    _, _, SparsePauliOp = _import_qiskit()

    x_ops: list[SparsePauliOp] = []
    y_ops: list[SparsePauliOp] = []
    z_ops: list[SparsePauliOp] = []
    for site in range(L):
        px = ["I"] * L
        py = ["I"] * L
        pz = ["I"] * L
        px[site] = "X"
        py[site] = "Y"
        pz[site] = "Z"
        x_ops.append(SparsePauliOp.from_list([("".join(px), 1.0)]))
        y_ops.append(SparsePauliOp.from_list([("".join(py), 1.0)]))
        z_ops.append(SparsePauliOp.from_list([("".join(pz), 1.0)]))
    return x_ops, y_ops, z_ops


def single_state_observables_qiskit(
    state: np.ndarray,
    L: int,
    ops: tuple[list["SparsePauliOp"], list["SparsePauliOp"], list["SparsePauliOp"]] | None = None,
) -> np.ndarray:
    _, Statevector, _ = _import_qiskit()
    if state.shape != (2**L,):
        raise ValueError("state shape must be (2**L,)")

    x_ops, y_ops, z_ops = ops if ops is not None else build_local_pauli_ops(L)
    sv = Statevector(state)

    out = np.zeros((L, 3), dtype=float)
    for site in range(L):
        out[site, 0] = float(np.real(sv.expectation_value(x_ops[site])))
        out[site, 1] = float(np.real(sv.expectation_value(y_ops[site])))
        out[site, 2] = float(np.real(sv.expectation_value(z_ops[site])))
    return out


def all_states_observables_qiskit(states: np.ndarray, L: int) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] != 2**L:
        raise ValueError("states shape must be (n_times, 2**L)")

    ops = build_local_pauli_ops(L)
    out = np.zeros((states.shape[0], L, 3), dtype=float)
    for idx, state in enumerate(states):
        out[idx] = single_state_observables_qiskit(state=state, L=L, ops=ops)
    return out


def run_qiskit_trotter(
    cfg: QiskitSimulationConfig,
    times: np.ndarray,
    order: int = 2,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    state0, bitstring, rotate_site = initial_statevector(
        L=cfg.L,
        init_pattern=cfg.init_pattern,
        phi=cfg.phi,
        rotate_site=cfg.rotate_site,
    )
    states = evolve_trotter_states_qiskit(
        state0=state0,
        L=cfg.L,
        Jz=cfg.Jz,
        boundary=cfg.boundary,
        times=times,
        order=order,
    )
    observables = all_states_observables_qiskit(states=states, L=cfg.L)
    return states, observables, bitstring, rotate_site


__all__ = [
    "Boundary",
    "InitPattern",
    "QiskitSimulationConfig",
    "all_states_observables_qiskit",
    "append_trotter_interval",
    "append_xxz_two_site_evolution",
    "build_bonds",
    "build_initial_bitstring",
    "build_initial_circuit",
    "build_local_pauli_ops",
    "build_time_evolution_circuit",
    "evolve_trotter_states_qiskit",
    "initial_statevector",
    "run_qiskit_trotter",
    "single_state_observables_qiskit",
]
