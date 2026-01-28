"""Entropic S-grid construction for MCMC model.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]

The S variable is the discrete entropic coordinate used throughout the model.
Two distinct grids are available:

1. Pre-BB grid (S ∈ [0, 1.001)): Pre-geometric regime
2. Post-BB grid (S ∈ [1.001, 95.07]): Observable cosmology regime

CRITICAL: S_BB = S_GEOM = 1.001 is the Big Bang observable, NOT "today".
S_0 ≈ 95.07 is today (calibrated with Ω_b = 0.0493).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from mcmc.core.ontology import THRESHOLDS, DS_CANONICAL, OntologicalThresholds


# Legacy alias for backwards compatibility
@dataclass(frozen=True)
class Seals:
    """Legacy: Ontological thresholds.

    DEPRECATED: Use OntologicalThresholds instead.

    Note: S4 = 1.001 is the Big Bang observable, NOT "today".
    """
    S1: float = THRESHOLDS.S_PLANCK  # 0.009
    S2: float = THRESHOLDS.S_GUT     # 0.099
    S3: float = THRESHOLDS.S_EW      # 0.999
    S4: float = THRESHOLDS.S_BB      # 1.001 = Big Bang, NOT today


@dataclass
class SGrid:
    """Discrete S-grid for entropic integration."""
    S_min: float = 0.010
    S_max: float = 1.001  # Default: pre-BB regime up to Big Bang
    dS: float = DS_CANONICAL
    thresholds: OntologicalThresholds = field(default_factory=lambda: THRESHOLDS)

    # Legacy compatibility
    @property
    def seals(self) -> Seals:
        return Seals()

    def build(self) -> np.ndarray:
        """Build the S array."""
        S = np.arange(self.S_min, self.S_max + 0.5 * self.dS, self.dS)
        return S

    def assert_thresholds_on_grid(self, S: np.ndarray) -> None:
        """Verify that relevant thresholds are on the grid."""
        for name in ["S_PLANCK", "S_GUT", "S_EW", "S_BB"]:
            val = getattr(self.thresholds, name)
            if self.S_min <= val <= self.S_max:
                if not np.any(np.isclose(S, val, atol=1e-9)):
                    raise ValueError(
                        f"Threshold {name}={val} not on grid. Adjust dS or range."
                    )

    # Legacy alias
    def assert_seals_on_grid(self, S: np.ndarray) -> None:
        self.assert_thresholds_on_grid(S)


@dataclass
class PreBBGrid(SGrid):
    """Grid for pre-Big-Bang regime (primordial)."""
    S_min: float = 0.001
    S_max: float = THRESHOLDS.S_BB  # 1.001


@dataclass
class PostBBGrid(SGrid):
    """Grid for post-Big-Bang regime (observable cosmology).

    CORRECCIÓN: Post-Big Bang S ∈ [1.001, 95.07]

    Note: For post-BB, we typically work in (t, z) space rather than S.
    This grid is provided for completeness if S is used as global clock.
    """
    S_min: float = THRESHOLDS.S_BB  # 1.001
    S_max: float = 95.07  # S_0 (present cosmological epoch)


def create_default_grid() -> tuple[SGrid, np.ndarray]:
    """Create the default pre-BB grid."""
    grid = SGrid()
    S = grid.build()
    # Skip threshold check for legacy grids
    return grid, S


def create_prebb_grid(S_min: float = 0.001) -> tuple[PreBBGrid, np.ndarray]:
    """Create a pre-Big-Bang grid."""
    grid = PreBBGrid(S_min=S_min)
    S = grid.build()
    # Skip threshold check for legacy grids
    return grid, S


def create_postbb_grid(S_max: float = 95.07) -> tuple[PostBBGrid, np.ndarray]:
    """Create a post-Big-Bang grid.

    CORRECCIÓN: Default S_max now 95.07 (present cosmological epoch)
    """
    grid = PostBBGrid(S_max=S_max)
    S = grid.build()
    return grid, S
