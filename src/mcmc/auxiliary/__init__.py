"""Auxiliary modules for MCMC cosmology.

Additional physics modules that extend the core MCMC framework:
- baryogenesis: Baryogenesis and matter-antimatter asymmetry
"""
from __future__ import annotations

from mcmc.auxiliary.baryogenesis import (
    BaryogenesisParams,
    BaryogenesisModel,
    eta_B_of_S,
    sakharov_conditions,
    cp_violation_mcmc,
)

__all__ = [
    "BaryogenesisParams",
    "BaryogenesisModel",
    "eta_B_of_S",
    "sakharov_conditions",
    "cp_violation_mcmc",
]
