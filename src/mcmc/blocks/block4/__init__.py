"""Block 4: Lattice-Gauge Simulations for MCMC.

MCMC Ontology: This block implements lattice gauge theory simulations
to study the MCMC mass gap and non-perturbative QFT effects.

Key feature: The coupling beta(S) depends on the entropic coordinate,
allowing study of pre-geometric transitions.

Modules:
    - config: Configuration dataclasses
    - wilson_action: Wilson gauge action with beta(S)
    - monte_carlo: Metropolis and HMC samplers
    - correlators: Gluon and meson correlators
    - mass_gap: Mass gap extraction from correlators
    - s_scan: S-dependent scanning of the mass gap
"""
from __future__ import annotations

from mcmc.blocks.block4.config import (
    LatticeParams,
    WilsonParams,
    MonteCarloParams,
    MassGapParams,
)
from mcmc.blocks.block4.wilson_action import (
    WilsonAction,
    staple,
    plaquette,
    wilson_loop,
    beta_of_S,
)
from mcmc.blocks.block4.monte_carlo import (
    MetropolisSampler,
    HeatBathSampler,
    LatticeConfiguration,
    thermalize,
)
from mcmc.blocks.block4.correlators import (
    gluon_correlator,
    polyakov_loop,
    meson_correlator,
    CorrelatorData,
)
from mcmc.blocks.block4.mass_gap import (
    MassGapExtractor,
    fit_exponential_decay,
    extract_mass_gap,
)
from mcmc.blocks.block4.s_scan import (
    SScanAnalyzer,
    scan_mass_gap_vs_S,
    phase_transition_finder,
)

__all__ = [
    # Config
    "LatticeParams",
    "WilsonParams",
    "MonteCarloParams",
    "MassGapParams",
    # Wilson action
    "WilsonAction",
    "staple",
    "plaquette",
    "wilson_loop",
    "beta_of_S",
    # Monte Carlo
    "MetropolisSampler",
    "HeatBathSampler",
    "LatticeConfiguration",
    "thermalize",
    # Correlators
    "gluon_correlator",
    "polyakov_loop",
    "meson_correlator",
    "CorrelatorData",
    # Mass gap
    "MassGapExtractor",
    "fit_exponential_decay",
    "extract_mass_gap",
    # S-scan
    "SScanAnalyzer",
    "scan_mass_gap_vs_S",
    "phase_transition_finder",
]
