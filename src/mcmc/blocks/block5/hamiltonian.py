"""MCMC Hamiltonian for quantum simulation.

MCMC Ontology: The Hamiltonian encodes the MCMC dynamics:
- Kinetic term: Free evolution of S
- Potential term: S-dependent potential V(S) with transition at S_GEOM
- Interaction term: Coupling between S modes
- Decoherence: Lindblad terms for S-measurement
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from mcmc.blocks.block5.config import HilbertSpaceParams, HamiltonianParams
from mcmc.blocks.block5.hilbert_space import QuditState, QuditBasis, MultiQuditState
from mcmc.blocks.block5.operators import (
    TensorialOperator,
    creation_operator,
    annihilation_operator,
    number_operator,
)


def kinetic_term(
    params: HilbertSpaceParams | None = None,
    h_params: HamiltonianParams | None = None,
) -> TensorialOperator:
    """Kinetic energy term H_kin = omega * n.

    Args:
        params: Hilbert space parameters
        h_params: Hamiltonian parameters

    Returns:
        Kinetic Hamiltonian
    """
    if params is None:
        params = HilbertSpaceParams()
    if h_params is None:
        h_params = HamiltonianParams()

    n = number_operator(params)
    return h_params.omega * n


def potential_term(
    params: HilbertSpaceParams | None = None,
    h_params: HamiltonianParams | None = None,
) -> TensorialOperator:
    """Potential energy term V(S).

    V(S) = V_0 * [1 - tanh((S - S_trans) / width)]

    This creates a potential barrier at S_trans (Big Bang transition).

    Args:
        params: Hilbert space parameters
        h_params: Hamiltonian parameters

    Returns:
        Potential Hamiltonian
    """
    if params is None:
        params = HilbertSpaceParams()
    if h_params is None:
        h_params = HamiltonianParams()

    d = params.d
    basis = QuditBasis(params)
    S_trans = h_params.S_transition
    V_0 = h_params.V_amplitude
    width = 10.0  # Width of transition

    # Potential at each level
    V_diag = np.zeros(d)
    for k in range(d):
        S_k = basis.level_to_S(k)
        V_diag[k] = V_0 * (1 - np.tanh((S_k - S_trans) / width)) / 2

    matrix = np.diag(V_diag)
    return TensorialOperator(matrix, params, "V(S)")


def interaction_term(
    site1: int,
    site2: int,
    n_sites: int,
    params: HilbertSpaceParams | None = None,
    h_params: HamiltonianParams | None = None,
) -> np.ndarray:
    """Interaction term between two sites.

    H_int = g * (a_i^dag a_j + a_j^dag a_i)

    Args:
        site1, site2: Site indices
        n_sites: Total number of sites
        params: Hilbert space parameters
        h_params: Hamiltonian parameters

    Returns:
        Interaction Hamiltonian matrix
    """
    if params is None:
        params = HilbertSpaceParams()
    if h_params is None:
        h_params = HamiltonianParams()

    d = params.d
    g = h_params.g
    d ** n_sites

    # Build operators in tensor product space
    a = annihilation_operator(params).matrix
    a_dag = creation_operator(params).matrix
    I = np.eye(d)

    # Build a_i^dag a_j + h.c.
    def tensor_op(op_list):
        """Build tensor product of operators."""
        result = op_list[0]
        for op in op_list[1:]:
            result = np.kron(result, op)
        return result

    # a_i^dag a_j
    ops1 = [I] * n_sites
    ops1[site1] = a_dag
    ops1[site2] = a
    term1 = tensor_op(ops1)

    # a_j^dag a_i (hermitian conjugate)
    ops2 = [I] * n_sites
    ops2[site1] = a
    ops2[site2] = a_dag
    term2 = tensor_op(ops2)

    return g * (term1 + term2)


class MCMCHamiltonian:
    """MCMC Hamiltonian for quantum simulation."""

    def __init__(
        self,
        params: HilbertSpaceParams | None = None,
        h_params: HamiltonianParams | None = None,
    ):
        """Initialize Hamiltonian.

        Args:
            params: Hilbert space parameters
            h_params: Hamiltonian parameters
        """
        self.params = params or HilbertSpaceParams()
        self.h_params = h_params or HamiltonianParams()
        self._build_hamiltonian()

    def _build_hamiltonian(self) -> None:
        """Build the full Hamiltonian matrix."""
        # Single-site terms
        H_kin = kinetic_term(self.params, self.h_params)
        H_pot = potential_term(self.params, self.h_params)

        self.H_single = H_kin.matrix + H_pot.matrix
        self.H = self.H_single.copy()

        # Multi-site extension if needed
        n_sites = self.params.n_sites
        if n_sites > 1:
            d = self.params.d
            I = np.eye(d)

            # Tensor product of single-site Hamiltonians
            H_total = np.zeros((d ** n_sites, d ** n_sites), dtype=complex)
            for i in range(n_sites):
                ops = [I] * n_sites
                ops[i] = self.H_single
                H_i = ops[0]
                for op in ops[1:]:
                    H_i = np.kron(H_i, op)
                H_total += H_i

            # Add nearest-neighbor interactions
            for i in range(n_sites - 1):
                H_total += interaction_term(i, i + 1, n_sites, self.params, self.h_params)

            self.H = H_total

        # Build Lindblad operators if decoherence is included
        if self.h_params.include_decoherence:
            self._build_lindblad()

    def _build_lindblad(self) -> None:
        """Build Lindblad collapse operators."""
        gamma = self.h_params.decoherence_gamma
        d = self.params.d

        # Dephasing in S basis
        self.L_ops = []
        for k in range(d):
            L = np.zeros((d, d), dtype=complex)
            L[k, k] = np.sqrt(gamma)
            self.L_ops.append(L)

    @property
    def matrix(self) -> np.ndarray:
        """Hamiltonian matrix."""
        return self.H

    def eigenvalues(self) -> np.ndarray:
        """Compute energy eigenvalues."""
        return np.linalg.eigvalsh(self.H)

    def eigenstates(self) -> tuple[np.ndarray, list[QuditState]]:
        """Compute energy eigenvalues and eigenstates.

        Returns:
            Tuple of (eigenvalues, eigenstates)
        """
        vals, vecs = np.linalg.eigh(self.H)
        states = [QuditState(vecs[:, i], self.params.d, self.params) for i in range(self.params.d)]
        return vals, states

    def ground_state(self) -> tuple[float, QuditState]:
        """Get ground state energy and state.

        Returns:
            Tuple of (E_0, |psi_0>)
        """
        vals, states = self.eigenstates()
        return vals[0], states[0]

    def expect(self, state: QuditState | MultiQuditState) -> float:
        """Expectation value of Hamiltonian.

        Args:
            state: Quantum state

        Returns:
            Energy expectation value
        """
        if isinstance(state, MultiQuditState):
            amps = state.amplitudes
        else:
            amps = state.amplitudes
        return float(np.real(np.vdot(amps, self.H @ amps)))


def time_evolution(
    state: QuditState,
    H: MCMCHamiltonian | TensorialOperator | np.ndarray,
    t: float,
    n_steps: int = 1,
) -> QuditState:
    """Evolve state under Hamiltonian for time t.

    |psi(t)> = exp(-i*H*t) |psi(0)>

    Args:
        state: Initial state
        H: Hamiltonian
        t: Evolution time
        n_steps: Number of Trotter steps

    Returns:
        Evolved state
    """
    if isinstance(H, MCMCHamiltonian):
        H_mat = H.matrix
    elif isinstance(H, TensorialOperator):
        H_mat = H.matrix
    else:
        H_mat = H

    dt = t / n_steps
    U_step = expm(-1j * H_mat * dt)

    amps = state.amplitudes.copy()
    for _ in range(n_steps):
        amps = U_step @ amps

    return QuditState(amps, state.d, state.params)


def lindblad_evolution(
    state: QuditState,
    H: MCMCHamiltonian,
    t: float,
    dt: float = 0.01,
) -> QuditState:
    """Evolve state under Lindblad master equation.

    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})

    Uses quantum trajectory (jump) method for pure state evolution.

    Args:
        state: Initial state
        H: MCMC Hamiltonian (with Lindblad operators)
        t: Evolution time
        dt: Time step

    Returns:
        Final state (may be collapsed)
    """
    n_steps = int(t / dt)
    amps = state.amplitudes.copy()

    # Effective non-Hermitian Hamiltonian
    H_eff = H.matrix.copy()
    if hasattr(H, 'L_ops'):
        for L in H.L_ops:
            H_eff -= 0.5j * L.conj().T @ L

    U_eff = expm(-1j * H_eff * dt)

    for _ in range(n_steps):
        # Non-unitary evolution
        amps = U_eff @ amps
        norm_sq = np.sum(np.abs(amps) ** 2)

        # Jump probability
        p_jump = 1 - norm_sq

        if np.random.random() < p_jump and hasattr(H, 'L_ops'):
            # Quantum jump
            # Choose which jump operator
            probs = []
            for L in H.L_ops:
                L_amps = L @ amps
                probs.append(np.sum(np.abs(L_amps) ** 2))
            probs = np.array(probs)
            if np.sum(probs) > 0:
                probs /= np.sum(probs)
                j = np.random.choice(len(H.L_ops), p=probs)
                amps = H.L_ops[j] @ amps
        else:
            # Normalize
            if norm_sq > 1e-15:
                amps /= np.sqrt(norm_sq)

    return QuditState(amps, state.d, state.params)


def adiabatic_evolution(
    state: QuditState,
    H_initial: MCMCHamiltonian | np.ndarray,
    H_final: MCMCHamiltonian | np.ndarray,
    T: float,
    n_steps: int = 100,
) -> QuditState:
    """Adiabatic evolution from H_initial to H_final.

    H(t) = (1 - t/T) * H_initial + (t/T) * H_final

    Args:
        state: Initial state
        H_initial: Initial Hamiltonian
        H_final: Final Hamiltonian
        T: Total evolution time
        n_steps: Number of time steps

    Returns:
        Final state
    """
    if isinstance(H_initial, MCMCHamiltonian):
        H0 = H_initial.matrix
    else:
        H0 = H_initial

    if isinstance(H_final, MCMCHamiltonian):
        H1 = H_final.matrix
    else:
        H1 = H_final

    dt = T / n_steps
    amps = state.amplitudes.copy()

    for i in range(n_steps):
        s = (i + 0.5) / n_steps  # Midpoint rule
        H_t = (1 - s) * H0 + s * H1
        U_step = expm(-1j * H_t * dt)
        amps = U_step @ amps

    return QuditState(amps, state.d, state.params)


class TimeEvolver:
    """Time evolution manager for MCMC quantum simulation."""

    def __init__(
        self,
        H: MCMCHamiltonian,
        dt: float = 0.01,
        use_lindblad: bool = True,
    ):
        """Initialize evolver.

        Args:
            H: MCMC Hamiltonian
            dt: Time step
            use_lindblad: Use Lindblad evolution
        """
        self.H = H
        self.dt = dt
        self.use_lindblad = use_lindblad

        # Pre-compute evolution operator for efficiency
        if not use_lindblad:
            self.U_dt = expm(-1j * H.matrix * dt)

    def step(self, state: QuditState) -> QuditState:
        """Evolve state by one time step.

        Args:
            state: Input state

        Returns:
            Evolved state
        """
        if self.use_lindblad:
            return lindblad_evolution(state, self.H, self.dt, self.dt)
        else:
            new_amps = self.U_dt @ state.amplitudes
            return QuditState(new_amps, state.d, state.params)

    def evolve(
        self,
        state: QuditState,
        t: float,
        record_trajectory: bool = False,
    ) -> QuditState | tuple[QuditState, list[QuditState]]:
        """Evolve state for time t.

        Args:
            state: Initial state
            t: Total evolution time
            record_trajectory: Whether to record intermediate states

        Returns:
            Final state (and trajectory if requested)
        """
        n_steps = int(t / self.dt)
        trajectory = [state.copy()] if record_trajectory else None

        current = state.copy()
        for _ in range(n_steps):
            current = self.step(current)
            if record_trajectory:
                trajectory.append(current.copy())

        if record_trajectory:
            return current, trajectory
        return current
