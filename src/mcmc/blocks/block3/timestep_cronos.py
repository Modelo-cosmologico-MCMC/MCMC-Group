"""Cronos timestep integrator for N-body simulations.

MCMC Ontology: The Cronos timestep dt_C = dt_N * N(S) incorporates
the entropic lapse function N(S) = exp(Phi_ten(S)).

Key features:
- In early universe (low S near S_GEOM), N(S) > 1, time runs faster
- In present (S ~ S_0), N(S) ~ 1, standard dynamics
- Stratified present: N_local varies with local S(x)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Callable

from mcmc.core.ontology import THRESHOLDS, S_0, S_GEOM
from mcmc.blocks.block3.config import TimestepCronosParams


@dataclass
class TimestepState:
    """State of the timestep controller.

    Attributes:
        dt_current: Current timestep
        N_current: Current lapse value
        S_current: Current entropic coordinate
        step_count: Number of steps taken
        time_elapsed: Total time elapsed
    """
    dt_current: float
    N_current: float
    S_current: float
    step_count: int = 0
    time_elapsed: float = 0.0


def lapse_function_default(S: float | np.ndarray) -> float | np.ndarray:
    """Default entropic lapse function N(S).

    N(S) = exp(Phi_ten(S)) where Phi_ten is the tensional phase.

    For the geometric regime (S >= S_GEOM):
    - N(S) approaches 1 as S -> S_0 (present)
    - N(S) > 1 for S << S_0 (early universe, faster dynamics)

    Args:
        S: Entropic coordinate(s)

    Returns:
        Lapse function N(S)
    """
    is_scalar = np.isscalar(S)
    S_arr = np.atleast_1d(np.asarray(S, dtype=float))

    # Phi_ten: Gaussian centered at S_STAR_PEAK with smooth transition
    S_peak = THRESHOLDS.S_STAR_PEAK  # ~47.5
    sigma = 20.0  # Width of transition

    phi_ten = 0.3 * np.exp(-0.5 * ((S_arr - S_peak) / sigma) ** 2)

    # Add small correction for very early times (near S_GEOM)
    early_factor = np.exp(-0.1 * (S_arr - S_GEOM))
    early_mask = S_arr < 10.0
    phi_ten[early_mask] += 0.2 * early_factor[early_mask]

    N = np.exp(phi_ten)
    return float(N[0]) if is_scalar else N


class CronosTimestep:
    """Cronos timestep controller for N-body integration.

    The Cronos timestep incorporates the entropic lapse:
        dt_Cronos = dt_Newton * N(S)

    This ensures that:
    1. Dynamics respect the MCMC ontology (time flows via Cronos law)
    2. Early universe has modified time evolution
    3. Present approaches standard Newtonian dynamics
    """

    def __init__(
        self,
        params: TimestepCronosParams | None = None,
        lapse_func: Callable[[float], float] | None = None,
    ):
        """Initialize the Cronos timestep controller.

        Args:
            params: Configuration parameters
            lapse_func: Custom lapse function N(S). If None, uses default.
        """
        self.params = params or TimestepCronosParams()
        self.lapse_func = lapse_func or lapse_function_default
        self.state = TimestepState(
            dt_current=self.params.dt_base,
            N_current=self.lapse_func(self.params.S_current),
            S_current=self.params.S_current,
        )

    def N(self, S: float | None = None) -> float:
        """Compute lapse function at given S.

        Args:
            S: Entropic coordinate. If None, uses current state.

        Returns:
            Lapse function value N(S)
        """
        if S is None:
            S = self.state.S_current
        N_val = self.lapse_func(S)
        return float(np.clip(N_val, self.params.N_min, self.params.N_max))

    def dt_cronos(
        self,
        dt_newton: float | None = None,
        S: float | None = None,
    ) -> float:
        """Compute Cronos timestep from Newtonian timestep.

        dt_Cronos = dt_Newton * N(S)

        Args:
            dt_newton: Newtonian timestep. If None, uses dt_base.
            S: Entropic coordinate. If None, uses current state.

        Returns:
            Cronos timestep
        """
        if dt_newton is None:
            dt_newton = self.params.dt_base
        N_val = self.N(S)
        return dt_newton * N_val

    def compute_cfl_timestep(
        self,
        dx: float,
        v_max: float,
        S: float | None = None,
    ) -> float:
        """Compute CFL-limited Cronos timestep.

        The CFL condition is applied to the Cronos timestep:
            dt_Cronos <= cfl_factor * dx / v_max

        This becomes:
            dt_Newton <= cfl_factor * dx / (v_max * N(S))

        Args:
            dx: Minimum grid spacing
            v_max: Maximum velocity in simulation
            S: Entropic coordinate

        Returns:
            CFL-limited timestep
        """
        N_val = self.N(S)
        v_max = max(v_max, 1e-10)  # Avoid division by zero
        dt_cfl = self.params.cfl_factor * dx / v_max
        dt_cronos = dt_cfl * N_val
        return min(dt_cronos, self.params.dt_base * N_val)

    def step(
        self,
        dt_requested: float | None = None,
        dx: float | None = None,
        v_max: float | None = None,
    ) -> float:
        """Advance timestep state and return timestep to use.

        Args:
            dt_requested: Requested Newton timestep
            dx: Grid spacing for CFL (optional)
            v_max: Max velocity for CFL (optional)

        Returns:
            Actual Cronos timestep to use
        """
        dt_newton = dt_requested or self.params.dt_base

        if self.params.adapt_cfl and dx is not None and v_max is not None:
            dt_cronos = self.compute_cfl_timestep(dx, v_max)
        else:
            dt_cronos = self.dt_cronos(dt_newton)

        # Update state
        self.state.dt_current = dt_cronos
        self.state.N_current = self.N()
        self.state.step_count += 1
        self.state.time_elapsed += dt_cronos

        return dt_cronos

    def update_S(self, S_new: float) -> None:
        """Update the current entropic coordinate.

        Args:
            S_new: New entropic coordinate
        """
        self.state.S_current = S_new
        self.state.N_current = self.N(S_new)

    def reset(self) -> None:
        """Reset timestep state to initial conditions."""
        self.state = TimestepState(
            dt_current=self.params.dt_base,
            N_current=self.lapse_func(self.params.S_current),
            S_current=self.params.S_current,
        )


def create_adaptive_timestep(
    S_current: float = S_0,
    dt_base: float = 0.01,
    cfl_factor: float = 0.4,
) -> CronosTimestep:
    """Factory function for adaptive Cronos timestep.

    Args:
        S_current: Current entropic coordinate
        dt_base: Base timestep
        cfl_factor: CFL safety factor

    Returns:
        CronosTimestep with adaptive CFL control
    """
    params = TimestepCronosParams(
        S_current=S_current,
        dt_base=dt_base,
        adapt_cfl=True,
        cfl_factor=cfl_factor,
    )
    return CronosTimestep(params)


def create_fixed_timestep(
    S_current: float = S_0,
    dt_fixed: float = 0.01,
) -> CronosTimestep:
    """Factory function for fixed Cronos timestep.

    Args:
        S_current: Current entropic coordinate
        dt_fixed: Fixed timestep

    Returns:
        CronosTimestep with fixed timestep (no CFL adaptation)
    """
    params = TimestepCronosParams(
        S_current=S_current,
        dt_base=dt_fixed,
        adapt_cfl=False,
    )
    return CronosTimestep(params)
