"""Analysis modules for MCMC N-body simulations.

These modules analyze simulation outputs and compare with observations:
- rotation_curves: Galaxy rotation curve analysis
- mass_function: Halo mass function
- subhalo_count: Subhalo abundance
- sparc_comparison: Comparison with SPARC database
"""
from __future__ import annotations

from mcmc.blocks.block3.analysis.rotation_curves import (
    RotationCurveAnalyzer,
    RotationCurveParams,
    compute_rotation_curve,
    fit_rotation_curve,
)
from mcmc.blocks.block3.analysis.mass_function import (
    HaloMassFunction,
    MassFunctionParams,
    press_schechter,
    sheth_tormen,
    compute_mass_function_mcmc,
)
from mcmc.blocks.block3.analysis.subhalo_count import (
    SubhaloCounter,
    SubhaloParams,
    count_subhalos,
    subhalo_mass_function,
)
from mcmc.blocks.block3.analysis.sparc_comparison import (
    SPARCComparison,
    load_sparc_galaxy,
    compare_mcmc_to_sparc,
)

__all__ = [
    "RotationCurveAnalyzer",
    "RotationCurveParams",
    "compute_rotation_curve",
    "fit_rotation_curve",
    "HaloMassFunction",
    "MassFunctionParams",
    "press_schechter",
    "sheth_tormen",
    "compute_mass_function_mcmc",
    "SubhaloCounter",
    "SubhaloParams",
    "count_subhalos",
    "subhalo_mass_function",
    "SPARCComparison",
    "load_sparc_galaxy",
    "compare_mcmc_to_sparc",
]
