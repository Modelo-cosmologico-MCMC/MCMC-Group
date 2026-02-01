"""Tensorial operators for MCMC quantum simulation.

MCMC Ontology: Key operators include:
- S operator: Measures entropic coordinate
- Creation/annihilation: Raise/lower S
- Collapse operator: Induces S-measurement/decoherence
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from mcmc.blocks.block5.config import HilbertSpaceParams, OperatorParams
from mcmc.blocks.block5.hilbert_space import QuditState, QuditBasis


class TensorialOperator:
    """Base class for tensorial operators."""

    def __init__(
        self,
        matrix: np.ndarray,
        params: HilbertSpaceParams | None = None,
        name: str = "Operator",
    ):
        """Initialize operator.

        Args:
            matrix: Operator matrix
            params: Hilbert space parameters
            name: Operator name
        """
        self.matrix = np.asarray(matrix, dtype=complex)
        self.params = params or HilbertSpaceParams()
        self.name = name
        self.d = self.matrix.shape[0]

    def __call__(self, state: QuditState) -> QuditState:
        """Apply operator to state.

        Args:
            state: Input state

        Returns:
            Output state O|psi>
        """
        new_amps = self.matrix @ state.amplitudes
        return QuditState(new_amps, state.d, state.params)

    def __matmul__(self, other: "TensorialOperator") -> "TensorialOperator":
        """Operator multiplication."""
        return TensorialOperator(
            self.matrix @ other.matrix,
            self.params,
            f"{self.name}*{other.name}",
        )

    def __add__(self, other: "TensorialOperator") -> "TensorialOperator":
        """Operator addition."""
        return TensorialOperator(
            self.matrix + other.matrix,
            self.params,
            f"({self.name}+{other.name})",
        )

    def __mul__(self, scalar: complex) -> "TensorialOperator":
        """Scalar multiplication."""
        return TensorialOperator(
            scalar * self.matrix,
            self.params,
            f"{scalar}*{self.name}",
        )

    def __rmul__(self, scalar: complex) -> "TensorialOperator":
        """Scalar multiplication (right)."""
        return self.__mul__(scalar)

    def dag(self) -> "TensorialOperator":
        """Hermitian conjugate."""
        return TensorialOperator(
            self.matrix.conj().T,
            self.params,
            f"{self.name}^dag",
        )

    def commutator(self, other: "TensorialOperator") -> "TensorialOperator":
        """Commutator [self, other] = self*other - other*self."""
        return TensorialOperator(
            self.matrix @ other.matrix - other.matrix @ self.matrix,
            self.params,
            f"[{self.name},{other.name}]",
        )

    def expect(self, state: QuditState) -> complex:
        """Expectation value <psi|O|psi>."""
        return complex(np.vdot(state.amplitudes, self.matrix @ state.amplitudes))

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        """Check if operator is Hermitian."""
        return bool(np.allclose(self.matrix, self.matrix.conj().T, atol=tol))

    def eigenvalues(self) -> np.ndarray:
        """Compute eigenvalues."""
        if self.is_hermitian():
            return np.linalg.eigvalsh(self.matrix)
        else:
            return np.linalg.eigvals(self.matrix)

    def eigenstates(self) -> tuple[np.ndarray, list[QuditState]]:
        """Compute eigenvalues and eigenstates.

        Returns:
            Tuple of (eigenvalues, eigenstates)
        """
        if self.is_hermitian():
            vals, vecs = np.linalg.eigh(self.matrix)
        else:
            vals, vecs = np.linalg.eig(self.matrix)

        states = [QuditState(vecs[:, i], self.d, self.params) for i in range(self.d)]
        return vals, states

    def to_sparse(self) -> sparse.csr_matrix:
        """Convert to sparse matrix."""
        return sparse.csr_matrix(self.matrix)


def creation_operator(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Creation (raising) operator a^dag.

    a^dag |n> = sqrt(n+1) |n+1>

    Raises S by one level.

    Args:
        params: Hilbert space parameters

    Returns:
        Creation operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    matrix = np.zeros((d, d), dtype=complex)
    for n in range(d - 1):
        matrix[n + 1, n] = np.sqrt(n + 1)

    return TensorialOperator(matrix, params, "a_dag")


def annihilation_operator(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Annihilation (lowering) operator a.

    a |n> = sqrt(n) |n-1>

    Lowers S by one level.

    Args:
        params: Hilbert space parameters

    Returns:
        Annihilation operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    matrix = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        matrix[n - 1, n] = np.sqrt(n)

    return TensorialOperator(matrix, params, "a")


def number_operator(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Number operator n = a^dag a.

    n |n> = n |n>

    Args:
        params: Hilbert space parameters

    Returns:
        Number operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    matrix = np.diag(np.arange(d, dtype=float))
    return TensorialOperator(matrix, params, "n")


def S_operator(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Entropic coordinate operator S.

    S |k> = S_k |k>

    where S_k = S_min + k * (S_max - S_min) / (d - 1)

    Args:
        params: Hilbert space parameters

    Returns:
        S operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d
    S_min, S_max = params.S_min, params.S_max

    S_values = S_min + np.arange(d) * (S_max - S_min) / (d - 1)
    matrix = np.diag(S_values)

    return TensorialOperator(matrix, params, "S")


def collapse_operator(
    params: HilbertSpaceParams | None = None,
    op_params: OperatorParams | None = None,
) -> TensorialOperator:
    """Collapse (Lindblad jump) operator for S-measurement.

    Models decoherence and collapse in the S basis.
    L = sqrt(gamma) * sum_k |k><k|

    Args:
        params: Hilbert space parameters
        op_params: Operator parameters

    Returns:
        Collapse operator
    """
    if params is None:
        params = HilbertSpaceParams()
    if op_params is None:
        op_params = OperatorParams()

    gamma = op_params.decoherence_rate

    # Dephasing operator (diagonal in S basis)
    S_op = S_operator(params)
    matrix = np.sqrt(gamma) * S_op.matrix / np.max(np.abs(S_op.matrix))

    return TensorialOperator(matrix, params, "L_collapse")


def pauli_x(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Generalized Pauli X (cyclic shift).

    X |k> = |k+1 mod d>

    Args:
        params: Hilbert space parameters

    Returns:
        Pauli X operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    matrix = np.zeros((d, d), dtype=complex)
    for k in range(d):
        matrix[(k + 1) % d, k] = 1.0

    return TensorialOperator(matrix, params, "X")


def pauli_z(params: HilbertSpaceParams | None = None) -> TensorialOperator:
    """Generalized Pauli Z (phase).

    Z |k> = omega^k |k>

    where omega = exp(2*pi*i/d)

    Args:
        params: Hilbert space parameters

    Returns:
        Pauli Z operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    omega = np.exp(2j * np.pi / d)
    matrix = np.diag([omega ** k for k in range(d)])

    return TensorialOperator(matrix, params, "Z")


def displacement_operator(
    alpha: complex,
    params: HilbertSpaceParams | None = None,
) -> TensorialOperator:
    """Displacement operator D(alpha) = exp(alpha*a^dag - alpha*a).

    Args:
        alpha: Displacement parameter
        params: Hilbert space parameters

    Returns:
        Displacement operator
    """
    if params is None:
        params = HilbertSpaceParams()

    a = annihilation_operator(params)
    a_dag = creation_operator(params)

    # Generator
    G = alpha * a_dag.matrix - np.conj(alpha) * a.matrix

    # Matrix exponential
    from scipy.linalg import expm
    matrix = expm(G)

    return TensorialOperator(matrix, params, f"D({alpha:.2f})")


def squeeze_operator(
    r: float,
    params: HilbertSpaceParams | None = None,
) -> TensorialOperator:
    """Squeezing operator S(r) = exp(r/2 * (a^2 - a^dag^2)).

    Args:
        r: Squeezing parameter
        params: Hilbert space parameters

    Returns:
        Squeezing operator
    """
    if params is None:
        params = HilbertSpaceParams()

    a = annihilation_operator(params)
    a_dag = creation_operator(params)

    # Generator
    G = 0.5 * r * (a.matrix @ a.matrix - a_dag.matrix @ a_dag.matrix)

    from scipy.linalg import expm
    matrix = expm(G)

    return TensorialOperator(matrix, params, f"S({r:.2f})")


def projection_operator(
    k: int,
    params: HilbertSpaceParams | None = None,
) -> TensorialOperator:
    """Projection operator onto level k.

    P_k = |k><k|

    Args:
        k: Level to project onto
        params: Hilbert space parameters

    Returns:
        Projection operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d

    if not 0 <= k < d:
        raise ValueError(f"Level {k} out of range [0, {d})")

    matrix = np.zeros((d, d), dtype=complex)
    matrix[k, k] = 1.0

    return TensorialOperator(matrix, params, f"P_{k}")


def S_range_projector(
    S_min_range: float,
    S_max_range: float,
    params: HilbertSpaceParams | None = None,
) -> TensorialOperator:
    """Projector onto S range [S_min_range, S_max_range].

    P = sum_{k: S_k in range} |k><k|

    Args:
        S_min_range: Minimum S
        S_max_range: Maximum S
        params: Hilbert space parameters

    Returns:
        Range projection operator
    """
    if params is None:
        params = HilbertSpaceParams()
    d = params.d
    basis = QuditBasis(params)

    matrix = np.zeros((d, d), dtype=complex)
    for k in range(d):
        S_k = basis.level_to_S(k)
        if S_min_range <= S_k <= S_max_range:
            matrix[k, k] = 1.0

    return TensorialOperator(matrix, params, f"P_S[{S_min_range:.1f},{S_max_range:.1f}]")
