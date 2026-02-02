"""Monte Carlo sampling for lattice gauge theory.

Implements Metropolis, heat bath, and HMC algorithms for sampling
gauge field configurations.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Iterator

from mcmc.blocks.block4.config import MonteCarloParams
from mcmc.blocks.block4.wilson_action import (
    WilsonAction,
    LatticeConfiguration,
    staple,
    random_su_matrix,
)


@dataclass
class MCMCStats:
    """Statistics from Monte Carlo run.

    Attributes:
        acceptance_rate: Fraction of accepted updates
        plaquette_history: Average plaquette vs sweep
        action_history: Total action vs sweep
    """
    acceptance_rate: float
    plaquette_history: np.ndarray
    action_history: np.ndarray


def project_su_N(U: np.ndarray) -> np.ndarray:
    """Project matrix to SU(N).

    Uses polar decomposition followed by det normalization.

    Args:
        U: Complex NxN matrix

    Returns:
        SU(N) matrix
    """
    N = U.shape[0]
    # Gram-Schmidt orthogonalization via QR
    Q, R = np.linalg.qr(U)
    # Make determinant = 1
    det = np.linalg.det(Q)
    Q = Q / det ** (1 / N)
    return Q


class MetropolisSampler:
    """Metropolis algorithm for lattice gauge theory."""

    def __init__(
        self,
        action: WilsonAction,
        params: MonteCarloParams | None = None,
    ):
        """Initialize Metropolis sampler.

        Args:
            action: Wilson action
            params: Monte Carlo parameters
        """
        self.action = action
        self.params = params or MonteCarloParams()
        self.rng = np.random.default_rng(self.params.seed)
        self.epsilon = 0.2  # Step size for SU(N) updates

    def propose_update(self, U: np.ndarray) -> np.ndarray:
        """Propose new link via small SU(N) perturbation.

        Args:
            U: Current link matrix

        Returns:
            Proposed new link
        """
        N = U.shape[0]
        dU = random_su_matrix(N, self.epsilon)
        return project_su_N(dU @ U)

    def update_link(
        self,
        config: LatticeConfiguration,
        site: tuple,
        mu: int,
    ) -> bool:
        """Attempt to update single link.

        Args:
            config: Lattice configuration
            site: Lattice site
            mu: Link direction

        Returns:
            True if update accepted
        """
        U_old = config.get_link(site, mu)
        S_old = self.action.local_action(config, site, mu)

        # Propose new link
        U_new = self.propose_update(U_old)
        config.set_link(site, mu, U_new)

        S_new = self.action.local_action(config, site, mu)

        # Metropolis accept/reject
        dS = S_new - S_old
        if dS < 0 or self.rng.random() < np.exp(-dS):
            return True
        else:
            # Reject: restore old link
            config.set_link(site, mu, U_old)
            return False

    def sweep(self, config: LatticeConfiguration) -> float:
        """Perform one sweep over all links.

        Args:
            config: Lattice configuration

        Returns:
            Acceptance rate for this sweep
        """
        accepted = 0
        total = 0

        lattice = self.action.lattice
        for x in range(lattice.Nx):
            for y in range(lattice.Ny):
                for z in range(lattice.Nz):
                    for t in range(lattice.Nt):
                        site = (x, y, z, t)
                        for mu in range(4):
                            if self.update_link(config, site, mu):
                                accepted += 1
                            total += 1

        return accepted / total

    def run(
        self,
        config: LatticeConfiguration,
        n_sweeps: int | None = None,
        measure_every: int | None = None,
    ) -> MCMCStats:
        """Run Monte Carlo simulation.

        Args:
            config: Initial configuration
            n_sweeps: Number of sweeps (default: params.n_sweeps)
            measure_every: Measurement frequency

        Returns:
            MCMCStats with results
        """
        if n_sweeps is None:
            n_sweeps = self.params.n_sweeps
        if measure_every is None:
            measure_every = self.params.n_skip

        plaquette_history = []
        action_history = []
        total_acceptance = 0.0

        for i in range(n_sweeps):
            acc = self.sweep(config)
            total_acceptance += acc

            if i % measure_every == 0:
                plaq = self.action.average_plaquette(config)
                act = self.action.total_action(config)
                plaquette_history.append(plaq)
                action_history.append(act)

        return MCMCStats(
            acceptance_rate=total_acceptance / n_sweeps,
            plaquette_history=np.array(plaquette_history),
            action_history=np.array(action_history),
        )


class HeatBathSampler:
    """Heat bath algorithm for SU(2) and SU(3).

    More efficient than Metropolis for gauge theories.
    """

    def __init__(
        self,
        action: WilsonAction,
        params: MonteCarloParams | None = None,
    ):
        """Initialize heat bath sampler.

        Args:
            action: Wilson action
            params: Monte Carlo parameters
        """
        self.action = action
        self.params = params or MonteCarloParams()
        self.rng = np.random.default_rng(self.params.seed)

    def su2_heat_bath(self, a: float) -> np.ndarray:
        """Generate SU(2) matrix with heat bath distribution.

        P(U) propto exp(a * Re Tr(U))

        Uses Kennedy-Pendleton algorithm.

        Args:
            a: Coupling parameter (= beta * |staple|)

        Returns:
            SU(2) matrix
        """
        # Kennedy-Pendleton algorithm for SU(2)
        if a < 0.5:
            # Low coupling: uniform sampling
            theta = self.rng.uniform(0, 2 * np.pi)
            phi = self.rng.uniform(0, np.pi)
            x0 = np.cos(theta) * np.sin(phi)
            x1 = np.sin(theta) * np.sin(phi)
            x2 = np.cos(phi)
            x3 = self.rng.uniform(-1, 1)
        else:
            # High coupling: biased sampling
            while True:
                x0 = self.rng.uniform(-1, 1)
                # Avoid overflow in exp(2*a*x0) by comparing logarithms instead
                u = self.rng.random()
                # rhs = log(sqrt(1 - x0^2)) + 2*a*x0 = 0.5*log1p(-x0^2) + 2*a*x0
                rhs = 0.5 * np.log1p(-x0**2) + 2 * a * x0
                if np.log(u) < rhs:
                    break

            phi = self.rng.uniform(0, 2 * np.pi)
            theta = np.arccos(self.rng.uniform(-1, 1))

            r = np.sqrt(1 - x0**2)
            x1 = r * np.sin(theta) * np.cos(phi)
            x2 = r * np.sin(theta) * np.sin(phi)
            x3 = r * np.cos(theta)

        # Construct SU(2) matrix
        U = np.array([
            [x0 + 1j * x3, x2 + 1j * x1],
            [-x2 + 1j * x1, x0 - 1j * x3]
        ], dtype=complex)

        return U

    def su3_heat_bath(self, staple_sum: np.ndarray, beta: float) -> np.ndarray:
        """Generate SU(3) matrix using Cabibbo-Marinari algorithm.

        Updates SU(3) through three SU(2) subgroups.

        Args:
            staple_sum: Sum of staples
            beta: Gauge coupling

        Returns:
            SU(3) matrix
        """
        # Start from identity
        U = np.eye(3, dtype=complex)

        # Update through SU(2) subgroups
        for subgroup in [(0, 1), (0, 2), (1, 2)]:
            i, j = subgroup

            # Extract SU(2) submatrix of (U @ staple^dag)
            W = U @ staple_sum.conj().T
            a_sq = (np.abs(W[i, i] + W[j, j])**2 +
                   np.abs(W[i, j] - W[j, i])**2)
            a = np.sqrt(a_sq) if a_sq > 0 else 1.0

            # Generate SU(2) heat bath update
            V = self.su2_heat_bath(beta * a / 3)

            # Embed in SU(3)
            R = np.eye(3, dtype=complex)
            R[i, i] = V[0, 0]
            R[i, j] = V[0, 1]
            R[j, i] = V[1, 0]
            R[j, j] = V[1, 1]

            U = R @ U

        return U

    def update_link(
        self,
        config: LatticeConfiguration,
        site: tuple,
        mu: int,
    ) -> None:
        """Update single link using heat bath.

        Args:
            config: Lattice configuration
            site: Lattice site
            mu: Link direction
        """
        beta = self.action.beta()
        S = staple(config, site, mu)
        N = config.N

        if N == 2:
            # SU(2) heat bath
            det_S = np.linalg.det(S)
            if np.abs(det_S) > 1e-10:
                a = beta * np.sqrt(np.abs(det_S))
                V = self.su2_heat_bath(a)
                # New link: V @ S^dag / |S|
                S_dag_norm = S.conj().T / np.sqrt(np.abs(det_S))
                U_new = V @ S_dag_norm
            else:
                U_new = random_su_matrix(2, 1.0)
        else:
            # SU(3) Cabibbo-Marinari
            U_new = self.su3_heat_bath(S, beta)

        config.set_link(site, mu, U_new)

    def sweep(self, config: LatticeConfiguration) -> None:
        """Perform one sweep over all links.

        Args:
            config: Lattice configuration
        """
        lattice = self.action.lattice
        for x in range(lattice.Nx):
            for y in range(lattice.Ny):
                for z in range(lattice.Nz):
                    for t in range(lattice.Nt):
                        site = (x, y, z, t)
                        for mu in range(4):
                            self.update_link(config, site, mu)

    def run(
        self,
        config: LatticeConfiguration,
        n_sweeps: int | None = None,
        measure_every: int | None = None,
    ) -> MCMCStats:
        """Run Monte Carlo simulation.

        Args:
            config: Initial configuration
            n_sweeps: Number of sweeps
            measure_every: Measurement frequency

        Returns:
            MCMCStats with results
        """
        if n_sweeps is None:
            n_sweeps = self.params.n_sweeps
        if measure_every is None:
            measure_every = self.params.n_skip

        plaquette_history = []
        action_history = []

        for i in range(n_sweeps):
            self.sweep(config)

            if i % measure_every == 0:
                plaq = self.action.average_plaquette(config)
                act = self.action.total_action(config)
                plaquette_history.append(plaq)
                action_history.append(act)

        return MCMCStats(
            acceptance_rate=1.0,  # Heat bath always accepts
            plaquette_history=np.array(plaquette_history),
            action_history=np.array(action_history),
        )


def thermalize(
    config: LatticeConfiguration,
    action: WilsonAction,
    n_therm: int | None = None,
    algorithm: str = "heatbath",
    params: MonteCarloParams | None = None,
) -> LatticeConfiguration:
    """Thermalize lattice configuration.

    Args:
        config: Initial configuration
        action: Wilson action
        n_therm: Number of thermalization sweeps
        algorithm: "metropolis" or "heatbath"
        params: Monte Carlo parameters

    Returns:
        Thermalized configuration
    """
    if params is None:
        params = MonteCarloParams()
    if n_therm is None:
        n_therm = params.n_thermalize

    if algorithm == "metropolis":
        sampler = MetropolisSampler(action, params)
    else:
        sampler = HeatBathSampler(action, params)

    for _ in range(n_therm):
        if isinstance(sampler, MetropolisSampler):
            sampler.sweep(config)
        else:
            sampler.sweep(config)

    return config


def configuration_generator(
    action: WilsonAction,
    params: MonteCarloParams | None = None,
    cold_start: bool = True,
) -> Iterator[LatticeConfiguration]:
    """Generator for thermalized configurations.

    Yields configurations after thermalization and between measurements.

    Args:
        action: Wilson action
        params: Monte Carlo parameters
        cold_start: Start from cold (identity) configuration

    Yields:
        Thermalized lattice configurations
    """
    if params is None:
        params = MonteCarloParams()

    # Initialize
    config = LatticeConfiguration(action.lattice, cold_start)
    sampler = HeatBathSampler(action, params)

    # Thermalize
    for _ in range(params.n_thermalize):
        sampler.sweep(config)

    # Generate configurations
    for _ in range(params.n_sweeps):
        for _ in range(params.n_skip):
            sampler.sweep(config)
        yield config
