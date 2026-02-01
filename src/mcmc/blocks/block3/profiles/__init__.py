"""Halo density profiles for MCMC N-body analysis.

Available profiles:
- NFW: Navarro-Frenk-White (standard CDM profile)
- Burkert: Cored profile (observed in dwarf galaxies)
- ZhaoMCMC: MCMC-modified profile with S-dependent core

All profiles support MCMC corrections via stratified present S_local.
"""
from __future__ import annotations

from mcmc.blocks.block3.profiles.nfw import (
    NFWProfile,
    NFWParams,
    nfw_density,
    nfw_mass,
    nfw_velocity,
)
from mcmc.blocks.block3.profiles.burkert import (
    BurkertProfile,
    BurkertParams,
    burkert_density,
    burkert_mass,
)
from mcmc.blocks.block3.profiles.zhao_mcmc import (
    ZhaoMCMCProfile,
    ZhaoMCMCParams,
    zhao_mcmc_density,
    zhao_mcmc_mass,
)

__all__ = [
    "NFWProfile",
    "NFWParams",
    "nfw_density",
    "nfw_mass",
    "nfw_velocity",
    "BurkertProfile",
    "BurkertParams",
    "burkert_density",
    "burkert_mass",
    "ZhaoMCMCProfile",
    "ZhaoMCMCParams",
    "zhao_mcmc_density",
    "zhao_mcmc_mass",
]
