"""Quantum gates for MCMC simulation.

Implements qudit gates for manipulating tensorial states.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from mcmc.blocks.block5.config import HilbertSpaceParams, GateParams
from mcmc.blocks.block5.hilbert_space import QuditState
from mcmc.blocks.block5.operators import (
    TensorialOperator,
    pauli_x,
    pauli_z,
    creation_operator,
    annihilation_operator,
    S_operator,
)


class QuantumGate(TensorialOperator):
    """Base class for quantum gates."""

    def __init__(
        self,
        matrix: np.ndarray,
        params: HilbertSpaceParams | None = None,
        name: str = "Gate",
    ):
        """Initialize gate."""
        super().__init__(matrix, params, name)
        # Verify unitarity
        if not self.is_unitary():
            # Force unitarity via polar decomposition
            U, _, Vh = np.linalg.svd(matrix)
            self.matrix = U @ Vh

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """Check if gate is unitary."""
        I = np.eye(self.d)
        return bool(np.allclose(self.matrix @ self.matrix.conj().T, I, atol=tol))


def hadamard_gate(params: HilbertSpaceParams | None = None) -> QuantumGate:
    """Generalized Hadamard gate (QFT).

    H = (1/sqrt(d)) * sum_jk omega^(jk) |j><k|

    Creates equal superposition from |0>.

    Args:
        params: Hilbert space parameters

    Returns:
        Hadamard gate
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    omega = np.exp(2j * np.pi / d)
    matrix = np.array([[omega ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)

    return QuantumGate(matrix, params, "H")


def phase_gate(theta: float, params: HilbertSpaceParams | None = None) -> QuantumGate:
    """Phase gate.

    P(theta) = diag(1, e^{i*theta}, e^{2i*theta}, ...)

    Args:
        theta: Phase angle
        params: Hilbert space parameters

    Returns:
        Phase gate
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    phases = np.exp(1j * theta * np.arange(d))
    matrix = np.diag(phases)

    return QuantumGate(matrix, params, f"P({theta:.2f})")


def rotation_gate(
    theta: float,
    axis: str = "z",
    params: HilbertSpaceParams | None = None,
) -> QuantumGate:
    """Rotation gate around specified axis.

    R_z(theta) = exp(-i * theta * Z / 2)
    R_x(theta) = exp(-i * theta * X / 2)

    Args:
        theta: Rotation angle
        axis: "x", "y", or "z"
        params: Hilbert space parameters

    Returns:
        Rotation gate
    """
    if params is None:
        params = HilbertSpaceParams()

    if axis == "z":
        gen = pauli_z(params).matrix
    elif axis == "x":
        gen = pauli_x(params).matrix
    else:  # y
        X = pauli_x(params).matrix
        Z = pauli_z(params).matrix
        gen = -1j * (X @ Z - Z @ X) / 2

    matrix = expm(-1j * theta * gen / 2)
    return QuantumGate(matrix, params, f"R_{axis}({theta:.2f})")


def S_rotation_gate(
    dS: float,
    params: HilbertSpaceParams | None = None,
) -> QuantumGate:
    """Rotation in S-space.

    Shifts S values by dS: |S> -> |S + dS>

    Args:
        dS: Amount to shift S
        params: Hilbert space parameters

    Returns:
        S-rotation gate
    """
    if params is None:
        params = HilbertSpaceParams()

    # Number of levels to shift
    S_step = (params.S_max - params.S_min) / (params.d - 1)
    n_shift = int(round(dS / S_step))

    # Cyclic shift matrix
    d = params.d
    matrix = np.zeros((d, d), dtype=complex)
    for k in range(d):
        k_new = (k + n_shift) % d
        matrix[k_new, k] = 1.0

    return QuantumGate(matrix, params, f"R_S({dS:.1f})")


def controlled_gate(
    gate: QuantumGate,
    n_qudits: int = 2,
    control: int = 0,
    target: int = 1,
) -> QuantumGate:
    """Controlled gate in multi-qudit space.

    Applies gate to target only if control is in highest level.

    Args:
        gate: Single-qudit gate to control
        n_qudits: Number of qudits
        control: Control qudit index
        target: Target qudit index

    Returns:
        Controlled gate
    """
    d = gate.d
    dim = d ** n_qudits
    matrix = np.eye(dim, dtype=complex)

    # For each control state |k>
    for k in range(d):
        if k == d - 1:  # Highest level activates gate
            # Apply gate to target
            for c_state in range(dim):
                # Check if control qudit is in state k
                idx = c_state
                c_val = (idx // (d ** (n_qudits - 1 - control))) % d
                if c_val == k:
                    # Get target state
                    t_val = (idx // (d ** (n_qudits - 1 - target))) % d
                    for t_new in range(d):
                        # New index with target changed
                        new_idx = idx - t_val * (d ** (n_qudits - 1 - target)) + t_new * (d ** (n_qudits - 1 - target))
                        matrix[new_idx, idx] = gate.matrix[t_new, t_val]

    return QuantumGate(matrix, gate.params, f"C-{gate.name}")


def mcmc_collapse_gate(
    S_target: float,
    width: float = 10.0,
    params: HilbertSpaceParams | None = None,
) -> QuantumGate:
    """MCMC collapse gate: projects toward S_target.

    Implements partial collapse/measurement in S basis.

    Args:
        S_target: Target S value
        width: Width of collapse region
        params: Hilbert space parameters

    Returns:
        Collapse gate (non-unitary, normalized)
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d
    S_min, S_max = params.S_min, params.S_max

    # Gaussian weighting centered at S_target
    S_values = S_min + np.arange(d) * (S_max - S_min) / (d - 1)
    weights = np.exp(-0.5 * ((S_values - S_target) / width) ** 2)
    weights = np.sqrt(weights)  # Amplitude weights

    # Diagonal matrix (not unitary)
    matrix = np.diag(weights)
    # Normalize to make trace-preserving on average
    matrix /= np.sqrt(np.sum(weights ** 2))

    return QuantumGate(matrix, params, f"Collapse({S_target:.1f})")


def circuit_layer(gates: list[QuantumGate]) -> QuantumGate:
    """Compose gates in parallel (tensor product).

    Args:
        gates: List of gates to apply in parallel

    Returns:
        Combined gate
    """
    if not gates:
        raise ValueError("Need at least one gate")

    matrix = gates[0].matrix
    for gate in gates[1:]:
        matrix = np.kron(matrix, gate.matrix)

    return QuantumGate(matrix, gates[0].params, "Layer")


def circuit_sequence(gates: list[QuantumGate]) -> QuantumGate:
    """Compose gates in sequence (matrix multiplication).

    Args:
        gates: List of gates to apply in sequence

    Returns:
        Combined gate
    """
    if not gates:
        raise ValueError("Need at least one gate")

    matrix = gates[0].matrix
    for gate in gates[1:]:
        matrix = gate.matrix @ matrix  # Apply right-to-left

    return QuantumGate(matrix, gates[0].params, "Sequence")


__all__ = [
    "QuantumGate",
    "hadamard_gate",
    "phase_gate",
    "rotation_gate",
    "S_rotation_gate",
    "controlled_gate",
    "mcmc_collapse_gate",
    "circuit_layer",
    "circuit_sequence",
]
