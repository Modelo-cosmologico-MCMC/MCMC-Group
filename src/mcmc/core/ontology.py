"""Ontological constants and thresholds for MCMC model.

According to the MCMC Master Manuscript (Zenodo), the entropic variable S
defines discrete thresholds corresponding to physical phase transitions:

- S_PLANCK ≈ 0.009: Planck scale (quantum-gravitational regime)
- S_GUT ≈ 0.099: Grand Unified Theory scale
- S_EW ≈ 0.999: Electroweak symmetry breaking
- S_BB ≈ 1.001: QCD confinement / Big Bang observable (4th heartbeat)

CRITICAL: S_BB = 1.001 is the "Big Bang observable", NOT "today".
The observable universe evolution occurs for S > S_BB.

Regimes:
- Pre-BB (S ∈ [0, S_BB]): Primordial regime, 4 collapses, constant fixing
- Post-BB (S > S_BB): Observable cosmology, accelerated expansion, structures
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OntologicalThresholds:
    """Ontological thresholds (critical S values).

    These correspond to physical phase transitions, NOT to cosmological
    normalization points.
    """
    # Planck scale - quantum gravity regime
    S_PLANCK: float = 0.009

    # GUT unification scale
    S_GUT: float = 0.099

    # Electroweak symmetry breaking
    S_EW: float = 0.999

    # QCD confinement / Big Bang observable (4th heartbeat)
    # CRITICAL: This is the Big Bang, NOT "today"
    S_BB: float = 1.001


@dataclass(frozen=True)
class Regimes:
    """Defines the two main regimes of the MCMC model."""
    # Pre-Big-Bang: primordial regime with 4 collapses
    PRE_BB_MIN: float = 0.0
    PRE_BB_MAX: float = 1.001  # S_BB

    # Post-Big-Bang: observable cosmology
    # S > S_BB corresponds to t > 0 (cosmic time after Big Bang)
    POST_BB_MIN: float = 1.001  # S_BB


# Canonical instances
THRESHOLDS = OntologicalThresholds()
REGIMES = Regimes()

# Discretization step (canonical)
DS_CANONICAL = 1e-3


def is_pre_bb(S: float) -> bool:
    """Check if S is in pre-Big-Bang regime."""
    return S <= THRESHOLDS.S_BB


def is_post_bb(S: float) -> bool:
    """Check if S is in post-Big-Bang regime."""
    return S > THRESHOLDS.S_BB
