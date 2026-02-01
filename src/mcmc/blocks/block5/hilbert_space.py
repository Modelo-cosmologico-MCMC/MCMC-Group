"""Qudit Hilbert space for MCMC quantum simulation.

MCMC Ontology: The entropic coordinate S is mapped to a qudit:
- S = 0 (primordial superposition) -> |0>
- S = S_GEOM (Big Bang) -> |k_geom>
- S = S_0 (present) -> |k_0>
- S = S_MAX -> |d-1>

The superposition of S values represents quantum indeterminacy
before collapse/measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Sequence

from mcmc.blocks.block5.config import HilbertSpaceParams


@dataclass
class QuditState:
    """Qudit quantum state.

    Attributes:
        amplitudes: Complex amplitudes in computational basis
        d: Local dimension
        params: Hilbert space parameters
    """
    amplitudes: np.ndarray
    d: int
    params: HilbertSpaceParams | None = None

    def __post_init__(self):
        """Validate and normalize state."""
        self.amplitudes = np.asarray(self.amplitudes, dtype=complex)
        if len(self.amplitudes) != self.d:
            raise ValueError(f"Amplitude length {len(self.amplitudes)} != dimension {self.d}")
        self.normalize()

    def normalize(self) -> None:
        """Normalize state to unit norm."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    @property
    def norm(self) -> float:
        """State norm."""
        return float(np.linalg.norm(self.amplitudes))

    def copy(self) -> "QuditState":
        """Return a copy of this state."""
        return QuditState(self.amplitudes.copy(), self.d, self.params)

    def inner(self, other: "QuditState") -> complex:
        """Inner product <self|other>."""
        return complex(np.vdot(self.amplitudes, other.amplitudes))

    def expect(self, operator: np.ndarray) -> complex:
        """Expectation value <psi|O|psi>."""
        return complex(np.vdot(self.amplitudes, operator @ self.amplitudes))

    def probabilities(self) -> np.ndarray:
        """Measurement probabilities in computational basis."""
        return np.abs(self.amplitudes) ** 2

    def measure(self) -> int:
        """Perform projective measurement, return outcome."""
        probs = self.probabilities()
        outcome = np.random.choice(self.d, p=probs)
        # Collapse state
        self.amplitudes = np.zeros(self.d, dtype=complex)
        self.amplitudes[outcome] = 1.0
        return outcome

    def S_value(self) -> float:
        """Expected S value from state.

        Maps qudit level k to S via:
            S = S_min + k * (S_max - S_min) / (d - 1)
        """
        if self.params is None:
            S_min, S_max = 0.0, 100.0
        else:
            S_min, S_max = self.params.S_min, self.params.S_max

        probs = self.probabilities()
        k_values = np.arange(self.d)
        k_mean = np.sum(k_values * probs)

        S = S_min + k_mean * (S_max - S_min) / (self.d - 1)
        return float(S)

    def coherence(self) -> float:
        """Quantum coherence (off-diagonal l1-norm).

        C = sum_{i!=j} |rho_{ij}|
        """
        rho = np.outer(self.amplitudes, np.conj(self.amplitudes))
        off_diag = np.abs(rho) - np.diag(np.diag(np.abs(rho)))
        return float(np.sum(off_diag))

    def entropy(self) -> float:
        """Von Neumann entropy (for pure states, this is 0)."""
        return 0.0  # Pure state

    def to_density_matrix(self) -> np.ndarray:
        """Convert to density matrix."""
        return np.outer(self.amplitudes, np.conj(self.amplitudes))


class QuditBasis:
    """Computational basis for qudit Hilbert space."""

    def __init__(self, params: HilbertSpaceParams | None = None):
        """Initialize basis.

        Args:
            params: Hilbert space parameters
        """
        self.params = params or HilbertSpaceParams()
        self.d = self.params.d

    def basis_state(self, k: int) -> QuditState:
        """Return computational basis state |k>.

        Args:
            k: Basis index (0 to d-1)

        Returns:
            Basis state |k>
        """
        if not 0 <= k < self.d:
            raise ValueError(f"Index {k} out of range [0, {self.d})")
        amps = np.zeros(self.d, dtype=complex)
        amps[k] = 1.0
        return QuditState(amps, self.d, self.params)

    def S_to_level(self, S: float) -> int:
        """Map entropic coordinate S to qudit level.

        Args:
            S: Entropic coordinate

        Returns:
            Qudit level k
        """
        S_min, S_max = self.params.S_min, self.params.S_max
        k = int(round((S - S_min) / (S_max - S_min) * (self.d - 1)))
        return max(0, min(self.d - 1, k))

    def level_to_S(self, k: int) -> float:
        """Map qudit level to entropic coordinate.

        Args:
            k: Qudit level

        Returns:
            S value
        """
        S_min, S_max = self.params.S_min, self.params.S_max
        return S_min + k * (S_max - S_min) / (self.d - 1)

    def all_levels(self) -> np.ndarray:
        """Return all S values for each level."""
        return np.array([self.level_to_S(k) for k in range(self.d)])


def create_vacuum(params: HilbertSpaceParams | None = None) -> QuditState:
    """Create vacuum state |0>.

    Args:
        params: Hilbert space parameters

    Returns:
        Vacuum state
    """
    if params is None:
        params = HilbertSpaceParams()
    amps = np.zeros(params.d, dtype=complex)
    amps[0] = 1.0
    return QuditState(amps, params.d, params)


def create_superposition(
    coefficients: Sequence[complex],
    params: HilbertSpaceParams | None = None,
) -> QuditState:
    """Create superposition state from coefficients.

    Args:
        coefficients: Amplitudes for each basis state
        params: Hilbert space parameters

    Returns:
        Superposition state
    """
    if params is None:
        params = HilbertSpaceParams()
    if len(coefficients) != params.d:
        raise ValueError(f"Need {params.d} coefficients, got {len(coefficients)}")
    return QuditState(np.array(coefficients, dtype=complex), params.d, params)


def create_coherent_state(
    alpha: complex,
    params: HilbertSpaceParams | None = None,
) -> QuditState:
    """Create coherent state |alpha> (truncated).

    |alpha> = exp(-|alpha|^2/2) sum_n (alpha^n / sqrt(n!)) |n>

    Args:
        alpha: Coherent state parameter
        params: Hilbert space parameters

    Returns:
        Coherent state
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d
    amps = np.zeros(d, dtype=complex)
    for n in range(d):
        # alpha^n / sqrt(n!)
        if n == 0:
            amps[n] = 1.0
        else:
            amps[n] = amps[n - 1] * alpha / np.sqrt(n)
    amps *= np.exp(-0.5 * np.abs(alpha) ** 2)
    return QuditState(amps, d, params)


def create_S_eigenstate(
    S: float,
    params: HilbertSpaceParams | None = None,
) -> QuditState:
    """Create eigenstate of S operator.

    Args:
        S: Target entropic coordinate
        params: Hilbert space parameters

    Returns:
        Basis state closest to S
    """
    if params is None:
        params = HilbertSpaceParams()
    basis = QuditBasis(params)
    k = basis.S_to_level(S)
    return basis.basis_state(k)


def tensor_product(
    state1: QuditState,
    state2: QuditState,
) -> QuditState:
    """Tensor product of two qudit states.

    Args:
        state1: First qudit state
        state2: Second qudit state

    Returns:
        Tensor product state
    """
    amps = np.kron(state1.amplitudes, state2.amplitudes)
    d_total = state1.d * state2.d
    return QuditState(amps, d_total)


class MultiQuditState:
    """State of multiple qudits.

    Represents a state in the tensor product Hilbert space
    H = H_1 ⊗ H_2 ⊗ ... ⊗ H_n
    """

    def __init__(
        self,
        amplitudes: np.ndarray,
        n_sites: int,
        d_per_site: int,
        params: HilbertSpaceParams | None = None,
    ):
        """Initialize multi-qudit state.

        Args:
            amplitudes: State vector (length d^n)
            n_sites: Number of qudits
            d_per_site: Local dimension per site
            params: Hilbert space parameters
        """
        self.n_sites = n_sites
        self.d = d_per_site
        self.dim = d_per_site ** n_sites
        self.params = params

        self.amplitudes = np.asarray(amplitudes, dtype=complex)
        if len(self.amplitudes) != self.dim:
            raise ValueError(f"State dimension {len(self.amplitudes)} != {self.dim}")
        self.normalize()

    def normalize(self) -> None:
        """Normalize state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 1e-15:
            self.amplitudes /= norm

    def partial_trace(self, keep_sites: list[int]) -> np.ndarray:
        """Compute partial trace (reduced density matrix).

        Args:
            keep_sites: Sites to keep

        Returns:
            Reduced density matrix
        """
        # Reshape state vector into tensor
        shape = [self.d] * self.n_sites
        self.amplitudes.reshape(shape)

        # Compute rho = |psi><psi|
        rho_shape = shape + shape
        rho = np.outer(self.amplitudes, np.conj(self.amplitudes)).reshape(rho_shape)

        # Trace out other sites
        trace_sites = [i for i in range(self.n_sites) if i not in keep_sites]
        for site in sorted(trace_sites, reverse=True):
            # Contract indices
            rho = np.trace(rho, axis1=site, axis2=site + self.n_sites)

        # Reshape to matrix
        kept_dim = self.d ** len(keep_sites)
        return rho.reshape(kept_dim, kept_dim)

    def entanglement_entropy(self, site: int) -> float:
        """Compute entanglement entropy of single site.

        Args:
            site: Site index

        Returns:
            Von Neumann entropy S = -Tr(rho log rho)
        """
        rho = self.partial_trace([site])
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def create_product_state(
    single_site_states: list[QuditState],
) -> MultiQuditState:
    """Create product state from single-site states.

    Args:
        single_site_states: List of single-site states

    Returns:
        Multi-qudit product state
    """
    if not single_site_states:
        raise ValueError("Need at least one state")

    amps = single_site_states[0].amplitudes
    d = single_site_states[0].d

    for state in single_site_states[1:]:
        if state.d != d:
            raise ValueError("All states must have same dimension")
        amps = np.kron(amps, state.amplitudes)

    return MultiQuditState(amps, len(single_site_states), d)
