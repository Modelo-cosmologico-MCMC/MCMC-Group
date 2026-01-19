"""Pre-Big-Bang solver: Ontological regime S ∈ [0, S_BB].

This solver handles the primordial regime where:
- Block 0: Pre-geometric state (S < S_PLANCK)
- Block 1: Cronos law integration (S_PLANCK to S_BB)

Output: Initial conditions for post-BB cosmology.

CRITICAL: This solver operates ONLY in the pre-BB regime (S ≤ S_BB = 1.001).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mcmc.core.ontology import THRESHOLDS
from mcmc.core.chronos import ChronosParams, t_of_S
from mcmc.blocks.block0 import run_block0, Block0Params
from mcmc.blocks.block1 import run_block1, Block1Params


@dataclass
class PreBBResult:
    """Result from pre-BB solver."""
    # Grid
    S: np.ndarray
    t: np.ndarray  # Chronos time (t=0 at S_BB)

    # Block 0 outputs (pre-geometric)
    Mp_pre: float
    Ep_pre: float
    phi_pre: float
    kpre_mean: float

    # Block 1 outputs (Cronos integrated)
    a: np.ndarray  # Scale factor (normalized: a(S_BB)=1)
    z: np.ndarray  # Redshift
    H_ref: np.ndarray  # Reference H(S)

    # Boundary conditions at S_BB (for post-BB)
    a_BB: float = 1.0  # a(S_BB) = 1 by normalization
    t_BB: float = 0.0  # t(S_BB) = 0 by Chronos anchor

    # Metadata
    params: dict = field(default_factory=dict)


@dataclass
class PreBBParams:
    """Parameters for pre-BB solver."""
    block0: Block0Params = field(default_factory=Block0Params)
    block1: Block1Params = field(default_factory=Block1Params)
    chronos: ChronosParams = field(default_factory=ChronosParams)


def solve_prebb(params: PreBBParams | None = None) -> PreBBResult:
    """Solve the pre-Big-Bang regime.

    Executes Block 0 (pre-geometric) and Block 1 (Cronos integration)
    to produce initial conditions for post-BB cosmology.

    Args:
        params: Pre-BB parameters. Uses defaults if None.

    Returns:
        PreBBResult with integrated quantities and boundary conditions.
    """
    if params is None:
        params = PreBBParams()

    # Block 0: Pre-geometric state
    b0 = run_block0(params.block0)

    # Block 1: Cronos integration
    b1 = run_block1(params.block1)

    # Convert lists to arrays
    S = np.array(b1["S"])
    a = np.array(b1["a"])
    z = np.array(b1["z"])
    H_ref = np.array(b1["H"])

    # Compute Chronos time with BB anchor
    t = t_of_S(S, params.chronos)

    # Verify anchor
    idx_bb = np.argmin(np.abs(S - THRESHOLDS.S_BB))
    t_BB = float(t[idx_bb])
    a_BB = float(a[idx_bb])

    return PreBBResult(
        S=S,
        t=t,
        Mp_pre=b0["Mp_pre"],
        Ep_pre=b0["Ep_pre"],
        phi_pre=b0["phi_pre"],
        kpre_mean=b0["kpre_mean"],
        a=a,
        z=z,
        H_ref=H_ref,
        a_BB=a_BB,
        t_BB=t_BB,
        params={
            "block0": params.block0,
            "block1": params.block1,
            "chronos": params.chronos,
        },
    )


def get_prebb_boundary_conditions(result: PreBBResult) -> dict[str, Any]:
    """Extract boundary conditions at S_BB for post-BB solver.

    These conditions provide the "handoff" between pre-BB and post-BB regimes.

    Returns:
        Dictionary with:
        - a_BB: Scale factor at Big Bang (normalized to 1)
        - t_BB: Time at Big Bang (= 0 by Chronos anchor)
        - Mp_pre, Ep_pre: Pre-geometric masses
        - phi_pre: Pre-geometric field value
    """
    return {
        "a_BB": result.a_BB,
        "t_BB": result.t_BB,
        "Mp_pre": result.Mp_pre,
        "Ep_pre": result.Ep_pre,
        "phi_pre": result.phi_pre,
        "kpre_mean": result.kpre_mean,
    }
