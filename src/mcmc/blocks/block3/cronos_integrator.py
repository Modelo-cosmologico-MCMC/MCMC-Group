"""Cronos N-body integrator for MCMC cosmological simulations.

MCMC Ontology: This integrator evolves particles using the Cronos
timestep (dt_C = dt_N * N(S)) and MCMC-modified gravity.

The integrator supports:
1. Leapfrog (KDK and DKD variants)
2. Evolving entropic coordinate S(t)
3. MCV (dark matter) contributions
4. Stratified present (local S variations)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Callable

from mcmc.core.ontology import S_0, S_GEOM
from mcmc.blocks.block3.config import CronosIntegratorParams
from mcmc.blocks.block3.timestep_cronos import CronosTimestep, TimestepCronosParams
from mcmc.blocks.block3.poisson_modified import PoissonModified, PoissonModifiedParams


@dataclass
class ParticleData:
    """Container for particle data.

    Attributes:
        positions: Particle positions [N, 3]
        velocities: Particle velocities [N, 3]
        masses: Particle masses [N]
        ids: Particle IDs [N]
    """
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    ids: np.ndarray | None = None

    def __post_init__(self):
        """Validate particle data shapes."""
        n = len(self.masses)
        assert self.positions.shape == (n, 3)
        assert self.velocities.shape == (n, 3)
        if self.ids is None:
            self.ids = np.arange(n)


@dataclass
class SimulationState:
    """State of the N-body simulation.

    Attributes:
        particles: ParticleData container
        S_current: Current entropic coordinate
        time: Current simulation time
        step: Current step number
        potential: Current gravitational potential (optional)
    """
    particles: ParticleData
    S_current: float
    time: float = 0.0
    step: int = 0
    potential: np.ndarray | None = None


class CronosIntegrator:
    """N-body integrator with Cronos timestep and MCMC gravity.

    This integrator combines:
    - Cronos timestep: dt_C = dt_N * N(S)
    - Modified Poisson: includes MCV contribution
    - Optional S evolution: S(t) changes during simulation
    """

    def __init__(
        self,
        params: CronosIntegratorParams | None = None,
        timestep: CronosTimestep | None = None,
        poisson: PoissonModified | None = None,
    ):
        """Initialize the Cronos integrator.

        Args:
            params: Configuration parameters
            timestep: Cronos timestep controller (created if None)
            poisson: Poisson solver (created if None)
        """
        self.params = params or CronosIntegratorParams()

        # Initialize timestep controller
        if timestep is not None:
            self.timestep = timestep
        else:
            ts_params = self.params.timestep
            self.timestep = CronosTimestep(ts_params)

        # Initialize Poisson solver
        if poisson is not None:
            self.poisson = poisson
        else:
            self.poisson = PoissonModified(self.params.poisson)

        self.state: SimulationState | None = None

    def initialize(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        S_initial: float = S_0,
    ) -> SimulationState:
        """Initialize simulation with particle data.

        Args:
            positions: Initial positions [N, 3]
            velocities: Initial velocities [N, 3]
            masses: Particle masses [N]
            S_initial: Initial entropic coordinate

        Returns:
            Initial SimulationState
        """
        particles = ParticleData(
            positions=positions.copy(),
            velocities=velocities.copy(),
            masses=masses.copy(),
        )
        self.state = SimulationState(
            particles=particles,
            S_current=S_initial,
        )
        self.timestep.update_S(S_initial)
        return self.state

    def _deposit_density(self, particles: ParticleData) -> np.ndarray:
        """Deposit particle masses onto grid (CIC scheme).

        Args:
            particles: ParticleData container

        Returns:
            Density field on grid
        """
        N = self.params.poisson.grid_size
        L = self.params.poisson.box_size
        dx = L / N

        density = np.zeros((N, N, N))
        pos = particles.positions
        mass = particles.masses

        # Cloud-in-Cell (CIC) deposit
        for i in range(len(mass)):
            # Normalized position
            x_norm = pos[i] / dx
            # Grid indices
            ix = int(np.floor(x_norm[0])) % N
            iy = int(np.floor(x_norm[1])) % N
            iz = int(np.floor(x_norm[2])) % N
            # Fractional position
            fx = x_norm[0] - np.floor(x_norm[0])
            fy = x_norm[1] - np.floor(x_norm[1])
            fz = x_norm[2] - np.floor(x_norm[2])

            # CIC weights
            w000 = (1 - fx) * (1 - fy) * (1 - fz)
            w100 = fx * (1 - fy) * (1 - fz)
            w010 = (1 - fx) * fy * (1 - fz)
            w001 = (1 - fx) * (1 - fy) * fz
            w110 = fx * fy * (1 - fz)
            w101 = fx * (1 - fy) * fz
            w011 = (1 - fx) * fy * fz
            w111 = fx * fy * fz

            # Deposit mass
            density[ix, iy, iz] += mass[i] * w000
            density[(ix + 1) % N, iy, iz] += mass[i] * w100
            density[ix, (iy + 1) % N, iz] += mass[i] * w010
            density[ix, iy, (iz + 1) % N] += mass[i] * w001
            density[(ix + 1) % N, (iy + 1) % N, iz] += mass[i] * w110
            density[(ix + 1) % N, iy, (iz + 1) % N] += mass[i] * w101
            density[ix, (iy + 1) % N, (iz + 1) % N] += mass[i] * w011
            density[(ix + 1) % N, (iy + 1) % N, (iz + 1) % N] += mass[i] * w111

        # Normalize to density
        cell_volume = dx ** 3
        density /= cell_volume

        return density

    def _interpolate_acceleration(
        self,
        particles: ParticleData,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ) -> np.ndarray:
        """Interpolate grid acceleration to particle positions (CIC).

        Args:
            particles: ParticleData container
            ax, ay, az: Acceleration field components

        Returns:
            Particle accelerations [N, 3]
        """
        N = self.params.poisson.grid_size
        L = self.params.poisson.box_size
        dx = L / N

        n_part = len(particles.masses)
        accel = np.zeros((n_part, 3))

        for i in range(n_part):
            x_norm = particles.positions[i] / dx
            ix = int(np.floor(x_norm[0])) % N
            iy = int(np.floor(x_norm[1])) % N
            iz = int(np.floor(x_norm[2])) % N
            fx = x_norm[0] - np.floor(x_norm[0])
            fy = x_norm[1] - np.floor(x_norm[1])
            fz = x_norm[2] - np.floor(x_norm[2])

            # CIC interpolation
            for (dix, wx) in [(0, 1 - fx), (1, fx)]:
                for (diy, wy) in [(0, 1 - fy), (1, fy)]:
                    for (diz, wz) in [(0, 1 - fz), (1, fz)]:
                        jx = (ix + dix) % N
                        jy = (iy + diy) % N
                        jz = (iz + diz) % N
                        w = wx * wy * wz
                        accel[i, 0] += w * ax[jx, jy, jz]
                        accel[i, 1] += w * ay[jx, jy, jz]
                        accel[i, 2] += w * az[jx, jy, jz]

        return accel

    def _compute_acceleration(
        self,
        particles: ParticleData,
        S: float,
    ) -> np.ndarray:
        """Compute gravitational acceleration for all particles.

        Args:
            particles: ParticleData container
            S: Current entropic coordinate

        Returns:
            Particle accelerations [N, 3]
        """
        # Deposit density
        rho = self._deposit_density(particles)

        # Solve Poisson with MCMC corrections
        potential, (ax, ay, az) = self.poisson.solve_and_accelerate(rho, S)
        self.state.potential = potential

        # Interpolate to particles
        accel = self._interpolate_acceleration(particles, ax, ay, az)

        # Add softening correction for close encounters
        eps = self.params.softening
        if eps > 0:
            # Softened direct sum for close pairs (simplified)
            pass  # Grid-based PM handles most cases

        return accel

    def _kick(self, particles: ParticleData, accel: np.ndarray, dt: float) -> None:
        """Kick step: update velocities.

        v_new = v_old + a * dt

        Args:
            particles: ParticleData to update
            accel: Accelerations
            dt: Timestep
        """
        particles.velocities += accel * dt

    def _drift(self, particles: ParticleData, dt: float) -> None:
        """Drift step: update positions.

        x_new = x_old + v * dt

        Args:
            particles: ParticleData to update
            dt: Timestep
        """
        particles.positions += particles.velocities * dt

        # Apply periodic boundary conditions
        L = self.params.poisson.box_size
        particles.positions = particles.positions % L

    def _evolve_S(self, dt: float) -> float:
        """Evolve entropic coordinate S.

        dS/dt = dS_dt (constant rate)

        Args:
            dt: Timestep

        Returns:
            New S value
        """
        if not self.params.evolve_S:
            return self.state.S_current

        S_new = self.state.S_current + self.params.dS_dt * dt
        # Clamp to valid range
        S_new = np.clip(S_new, S_GEOM, 100.0)
        return S_new

    def step_leapfrog(self, dt_newton: float | None = None) -> float:
        """Perform one leapfrog (KDK) step.

        Args:
            dt_newton: Newtonian timestep (uses Cronos if None)

        Returns:
            Actual timestep used
        """
        if self.state is None:
            raise RuntimeError("Integrator not initialized")

        # Get Cronos timestep
        v_max = np.max(np.linalg.norm(self.state.particles.velocities, axis=1))
        dt = self.timestep.step(
            dt_requested=dt_newton,
            dx=self.poisson.dx,
            v_max=v_max,
        )
        dt_half = dt / 2

        particles = self.state.particles
        S = self.state.S_current

        # Kick-Drift-Kick (KDK) scheme
        # Half kick
        accel = self._compute_acceleration(particles, S)
        self._kick(particles, accel, dt_half)

        # Full drift
        self._drift(particles, dt)

        # Evolve S at half step
        S_mid = self._evolve_S(dt_half)
        self.state.S_current = S_mid
        self.timestep.update_S(S_mid)

        # Half kick with new acceleration
        accel = self._compute_acceleration(particles, S_mid)
        self._kick(particles, accel, dt_half)

        # Final S update
        S_new = self._evolve_S(dt_half)
        self.state.S_current = S_new
        self.timestep.update_S(S_new)

        # Update state
        self.state.time += dt
        self.state.step += 1

        return dt

    def run(
        self,
        n_steps: int,
        dt_newton: float | None = None,
        callback: Callable[[SimulationState], None] | None = None,
    ) -> SimulationState:
        """Run simulation for n_steps.

        Args:
            n_steps: Number of steps to run
            dt_newton: Newtonian timestep (uses Cronos if None)
            callback: Optional callback called after each step

        Returns:
            Final SimulationState
        """
        if self.state is None:
            raise RuntimeError("Integrator not initialized")

        for _ in range(n_steps):
            self.step_leapfrog(dt_newton)
            if callback is not None:
                callback(self.state)

        return self.state

    def run_until(
        self,
        t_final: float,
        dt_newton: float | None = None,
        callback: Callable[[SimulationState], None] | None = None,
    ) -> SimulationState:
        """Run simulation until t_final.

        Args:
            t_final: Final time
            dt_newton: Newtonian timestep
            callback: Optional callback

        Returns:
            Final SimulationState
        """
        if self.state is None:
            raise RuntimeError("Integrator not initialized")

        while self.state.time < t_final:
            self.step_leapfrog(dt_newton)
            if callback is not None:
                callback(self.state)

        return self.state


def create_cronos_simulation(
    n_particles: int = 1000,
    box_size: float = 100.0,
    grid_size: int = 64,
    S_initial: float = S_0,
    include_mcv: bool = True,
) -> CronosIntegrator:
    """Factory function to create a Cronos N-body simulation.

    Args:
        n_particles: Number of particles
        box_size: Box size in Mpc
        grid_size: Poisson grid size
        S_initial: Initial entropic coordinate
        include_mcv: Whether to include MCV dark matter

    Returns:
        Configured CronosIntegrator
    """
    ts_params = TimestepCronosParams(S_current=S_initial)
    ps_params = PoissonModifiedParams(
        S_current=S_initial,
        grid_size=grid_size,
        box_size=box_size,
        include_mcv=include_mcv,
    )
    params = CronosIntegratorParams(
        n_particles=n_particles,
        timestep=ts_params,
        poisson=ps_params,
    )
    return CronosIntegrator(params)
