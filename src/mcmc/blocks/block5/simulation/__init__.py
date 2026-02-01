"""Qudit simulator for MCMC quantum dynamics.

Implements quantum trajectories, measurements, and observables.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import S_0
from mcmc.blocks.block5.config import (
    HamiltonianParams,
    SimulationParams,
)
from mcmc.blocks.block5.hilbert_space import (
    QuditState,
    create_vacuum,
    create_S_eigenstate,
)
from mcmc.blocks.block5.operators import (
    S_operator,
    number_operator,
)
from mcmc.blocks.block5.hamiltonian import (
    MCMCHamiltonian,
    TimeEvolver,
)


@dataclass
class TrajectoryResult:
    """Result of a single quantum trajectory.

    Attributes:
        times: Time points
        S_values: Expectation value of S at each time
        coherence: Coherence at each time
        probabilities: State probabilities at final time
        final_state: Final quantum state
        jumps: List of (time, jump_index) for quantum jumps
    """
    times: np.ndarray
    S_values: np.ndarray
    coherence: np.ndarray
    probabilities: np.ndarray
    final_state: QuditState
    jumps: list[tuple[float, int]]


@dataclass
class SimulationResult:
    """Result of ensemble simulation.

    Attributes:
        times: Time points
        S_mean: Mean S value at each time
        S_std: Standard deviation of S
        coherence_mean: Mean coherence
        n_trajectories: Number of trajectories
        final_distribution: Distribution of final S values
    """
    times: np.ndarray
    S_mean: np.ndarray
    S_std: np.ndarray
    coherence_mean: np.ndarray
    n_trajectories: int
    final_distribution: np.ndarray


class MCMCSimulator:
    """Quantum simulator for MCMC dynamics."""

    def __init__(
        self,
        params: SimulationParams | None = None,
    ):
        """Initialize simulator.

        Args:
            params: Simulation parameters
        """
        self.params = params or SimulationParams()
        self.h_params = self.params.hamiltonian
        self.hs_params = self.params.hilbert

        # Build Hamiltonian
        self.H = MCMCHamiltonian(self.hs_params, self.h_params)

        # Build evolver
        self.evolver = TimeEvolver(
            self.H,
            dt=self.params.dt,
            use_lindblad=self.h_params.include_decoherence,
        )

        # Observables
        self.S_op = S_operator(self.hs_params)
        self.n_op = number_operator(self.hs_params)

    def create_initial_state(
        self,
        state_type: str = "superposition",
        S_initial: float | None = None,
    ) -> QuditState:
        """Create initial state.

        Args:
            state_type: "vacuum", "S_eigenstate", "superposition", "thermal"
            S_initial: Initial S value (for eigenstate)

        Returns:
            Initial quantum state
        """
        if state_type == "vacuum":
            return create_vacuum(self.hs_params)
        elif state_type == "S_eigenstate":
            S = S_initial if S_initial is not None else S_0
            return create_S_eigenstate(S, self.hs_params)
        elif state_type == "superposition":
            # Equal superposition
            d = self.hs_params.d
            amps = np.ones(d, dtype=complex) / np.sqrt(d)
            return QuditState(amps, d, self.hs_params)
        elif state_type == "thermal":
            # Thermal state (Boltzmann weights)
            d = self.hs_params.d
            E = self.H.eigenvalues()
            beta = 1.0 / self.h_params.omega  # Temperature
            weights = np.exp(-beta * E)
            weights /= np.sum(weights)
            amps = np.sqrt(weights) * np.exp(1j * np.random.uniform(0, 2 * np.pi, d))
            return QuditState(amps, d, self.hs_params)
        else:
            raise ValueError(f"Unknown state type: {state_type}")

    def measure_S(self, state: QuditState) -> float:
        """Measure S expectation value.

        Args:
            state: Quantum state

        Returns:
            <S>
        """
        return float(np.real(self.S_op.expect(state)))

    def measure_coherence(self, state: QuditState) -> float:
        """Measure quantum coherence.

        Args:
            state: Quantum state

        Returns:
            Coherence (l1-norm of off-diagonal density matrix)
        """
        return state.coherence()

    def run_trajectory(
        self,
        initial_state: QuditState | None = None,
    ) -> TrajectoryResult:
        """Run single quantum trajectory.

        Args:
            initial_state: Initial state (creates superposition if None)

        Returns:
            TrajectoryResult
        """
        if initial_state is None:
            initial_state = self.create_initial_state("superposition")

        t_max = self.params.t_max
        dt = self.params.dt
        measure_interval = self.params.measure_interval

        n_steps = int(t_max / dt)
        measure_every = max(1, int(measure_interval / dt))

        # Storage
        times = []
        S_values = []
        coherence = []
        jumps = []

        state = initial_state.copy()
        t = 0.0

        for step in range(n_steps + 1):
            if step % measure_every == 0:
                times.append(t)
                S_values.append(self.measure_S(state))
                coherence.append(self.measure_coherence(state))

            if step < n_steps:
                state = self.evolver.step(state)
                t += dt

        return TrajectoryResult(
            times=np.array(times),
            S_values=np.array(S_values),
            coherence=np.array(coherence),
            probabilities=state.probabilities(),
            final_state=state,
            jumps=jumps,
        )

    def run_ensemble(
        self,
        n_trajectories: int | None = None,
        initial_state: QuditState | None = None,
    ) -> SimulationResult:
        """Run ensemble of trajectories.

        Args:
            n_trajectories: Number of trajectories
            initial_state: Initial state (same for all)

        Returns:
            SimulationResult with ensemble statistics
        """
        if n_trajectories is None:
            n_trajectories = self.params.n_trajectories

        # Run trajectories
        results = []
        for _ in range(n_trajectories):
            if initial_state is None:
                state = self.create_initial_state("superposition")
            else:
                state = initial_state.copy()
            results.append(self.run_trajectory(state))

        # Aggregate statistics
        times = results[0].times

        S_all = np.array([r.S_values for r in results])
        coherence_all = np.array([r.coherence for r in results])

        S_mean = np.mean(S_all, axis=0)
        S_std = np.std(S_all, axis=0)
        coherence_mean = np.mean(coherence_all, axis=0)

        # Final S distribution
        final_S = np.array([r.final_state.S_value() for r in results])
        final_distribution, _ = np.histogram(
            final_S,
            bins=self.hs_params.d,
            range=(self.hs_params.S_min, self.hs_params.S_max),
        )
        final_distribution = final_distribution / n_trajectories

        return SimulationResult(
            times=times,
            S_mean=S_mean,
            S_std=S_std,
            coherence_mean=coherence_mean,
            n_trajectories=n_trajectories,
            final_distribution=final_distribution,
        )

    def S_evolution_scan(
        self,
        S_initial_values: list[float],
    ) -> dict:
        """Scan S evolution from different initial S values.

        Args:
            S_initial_values: List of initial S values

        Returns:
            Dictionary with evolution results
        """
        results = {}
        for S_init in S_initial_values:
            state = create_S_eigenstate(S_init, self.hs_params)
            traj = self.run_trajectory(state)
            results[S_init] = {
                "times": traj.times,
                "S_evolution": traj.S_values,
                "coherence": traj.coherence,
                "final_S": traj.final_state.S_value(),
            }
        return results

    def collapse_dynamics(
        self,
        S_target: float,
        initial_state: QuditState | None = None,
        collapse_rate: float = 0.1,
    ) -> TrajectoryResult:
        """Simulate S-collapse dynamics toward target.

        Adds strong dephasing that drives S toward S_target.

        Args:
            S_target: Target S value for collapse
            initial_state: Initial superposition state
            collapse_rate: Rate of collapse/measurement

        Returns:
            TrajectoryResult showing collapse
        """
        # Create Hamiltonian with strong S-measuring decoherence
        h_params = HamiltonianParams(
            omega=self.h_params.omega,
            V_amplitude=0.0,  # No potential
            include_decoherence=True,
            decoherence_gamma=collapse_rate,
        )
        H_collapse = MCMCHamiltonian(self.hs_params, h_params)
        evolver = TimeEvolver(H_collapse, dt=self.params.dt, use_lindblad=True)

        if initial_state is None:
            initial_state = self.create_initial_state("superposition")

        # Run trajectory with modified evolver
        old_evolver = self.evolver
        self.evolver = evolver
        result = self.run_trajectory(initial_state)
        self.evolver = old_evolver

        return result


def mcmc_quantum_cosmology(
    S_primordial: float = 0.0,
    S_final: float = S_0,
    t_collapse: float = 10.0,
    params: SimulationParams | None = None,
) -> SimulationResult:
    """Simulate MCMC quantum cosmology from primordial to present.

    Models the collapse from primordial superposition (S~0) to
    present state (S~S_0).

    Args:
        S_primordial: Initial S (primordial superposition)
        S_final: Final S (present)
        t_collapse: Time scale for collapse
        params: Simulation parameters

    Returns:
        SimulationResult
    """
    if params is None:
        params = SimulationParams(t_max=t_collapse)

    sim = MCMCSimulator(params)

    # Start from primordial superposition
    initial_state = sim.create_initial_state("superposition")

    # Run ensemble
    result = sim.run_ensemble(initial_state=initial_state)

    return result


__all__ = [
    "TrajectoryResult",
    "SimulationResult",
    "MCMCSimulator",
    "mcmc_quantum_cosmology",
]
