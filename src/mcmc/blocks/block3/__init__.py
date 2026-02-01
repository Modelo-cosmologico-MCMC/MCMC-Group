"""Block 3: N-body Cronos Simulations.

MCMC Ontology: This block implements N-body simulations with
Cronos timestep integration and MCMC-modified gravity.

Modules:
    - config: Configuration dataclasses
    - timestep_cronos: Cronos timestep dt_C = dt_N * N(S)
    - poisson_modified: MCMC-modified Poisson solver
    - cronos_integrator: Full N-body integrator
    - profiles: Halo density profiles (NFW, Burkert, Zhao-MCMC)
    - analysis: Analysis routines (rotation curves, mass function)
"""
from __future__ import annotations

from mcmc.blocks.block3.config import (
    TimestepCronosParams,
    PoissonModifiedParams,
    CronosIntegratorParams,
    ProfileParams,
    AnalysisParams,
)
from mcmc.blocks.block3.timestep_cronos import (
    CronosTimestep,
    TimestepState,
    lapse_function_default,
    create_adaptive_timestep,
    create_fixed_timestep,
)
from mcmc.blocks.block3.poisson_modified import (
    PoissonModified,
    PoissonIsolated,
    PoissonState,
    rho_lat_simple,
    lambda_rel_simple,
    create_poisson_solver,
)
from mcmc.blocks.block3.cronos_integrator import (
    CronosIntegrator,
    ParticleData,
    SimulationState,
    create_cronos_simulation,
)

__all__ = [
    # Config
    "TimestepCronosParams",
    "PoissonModifiedParams",
    "CronosIntegratorParams",
    "ProfileParams",
    "AnalysisParams",
    # Timestep
    "CronosTimestep",
    "TimestepState",
    "lapse_function_default",
    "create_adaptive_timestep",
    "create_fixed_timestep",
    # Poisson
    "PoissonModified",
    "PoissonIsolated",
    "PoissonState",
    "rho_lat_simple",
    "lambda_rel_simple",
    "create_poisson_solver",
    # Integrator
    "CronosIntegrator",
    "ParticleData",
    "SimulationState",
    "create_cronos_simulation",
]
