"""Configuration dataclasses for Block 3: N-body Cronos.

MCMC Ontology: This block operates in the geometric regime S >= S_GEOM.
The Cronos timestep integrates N(S) from the entropic lapse function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mcmc.core.ontology import S_0


@dataclass(frozen=True)
class TimestepCronosParams:
    """Parameters for the Cronos timestep integrator.

    The timestep dt_Cronos = dt_Newt * N(S) where N(S) is the entropic lapse.
    This slows down dynamics in regions of low S (early universe).

    Attributes:
        S_current: Current entropic coordinate (default: S_0 = 95.07)
        dt_base: Base Newtonian timestep in code units
        N_min: Minimum allowed lapse (prevents runaway)
        N_max: Maximum allowed lapse
        adapt_cfl: Whether to apply CFL-like adaptive criterion
        cfl_factor: CFL safety factor (dt <= cfl_factor * dx / v_max)
    """
    S_current: float = S_0
    dt_base: float = 0.01
    N_min: float = 0.1
    N_max: float = 10.0
    adapt_cfl: bool = True
    cfl_factor: float = 0.4


@dataclass(frozen=True)
class PoissonModifiedParams:
    """Parameters for the MCMC-modified Poisson solver.

    The modified Poisson equation:
        nabla^2 Phi = 4*pi*G*rho_eff(S)

    where rho_eff includes contributions from MCV (dark matter) and
    stratified mass effects.

    Attributes:
        S_current: Current entropic coordinate
        include_mcv: Whether to include MCV (Masa Cuantica Virtual) contribution
        alpha_mcv: MCV coupling strength (fraction of rho_lat included)
        use_stratified: Whether to use stratified present (S_local varies)
        grid_size: Number of grid points per dimension
        box_size: Physical box size in Mpc
        boundary: Boundary condition type
    """
    S_current: float = S_0
    include_mcv: bool = True
    alpha_mcv: float = 1.0
    use_stratified: bool = False
    grid_size: int = 128
    box_size: float = 100.0  # Mpc
    boundary: Literal["periodic", "isolated"] = "periodic"


@dataclass(frozen=True)
class CronosIntegratorParams:
    """Parameters for the full Cronos N-body integrator.

    Combines timestep, Poisson, and particle evolution.

    Attributes:
        n_particles: Number of particles in simulation
        timestep: TimestepCronosParams instance
        poisson: PoissonModifiedParams instance
        softening: Gravitational softening length (code units)
        integrator: Integration scheme
        evolve_S: Whether S evolves during simulation
        dS_dt: Rate of S evolution if evolve_S is True
    """
    n_particles: int = 1000
    timestep: TimestepCronosParams = field(default_factory=TimestepCronosParams)
    poisson: PoissonModifiedParams = field(default_factory=PoissonModifiedParams)
    softening: float = 0.01
    integrator: Literal["leapfrog", "kdk", "dkd"] = "leapfrog"
    evolve_S: bool = False
    dS_dt: float = 0.0


@dataclass(frozen=True)
class ProfileParams:
    """Parameters for halo density profiles.

    Supports NFW, Burkert, and Zhao-MCMC profiles.

    Attributes:
        rho_s: Scale density [M_sun / Mpc^3]
        r_s: Scale radius [Mpc]
        r_vir: Virial radius [Mpc]
        concentration: Halo concentration c = r_vir / r_s
        S_halo: Entropic coordinate of halo (stratified present)
    """
    rho_s: float = 1e7  # M_sun / Mpc^3
    r_s: float = 0.02   # Mpc
    r_vir: float = 0.2  # Mpc
    concentration: float = 10.0
    S_halo: float = S_0


@dataclass(frozen=True)
class AnalysisParams:
    """Parameters for halo analysis routines.

    Attributes:
        r_min: Minimum radius for analysis [Mpc]
        r_max: Maximum radius for analysis [Mpc]
        n_bins: Number of radial bins
        mass_min: Minimum halo mass for mass function [M_sun]
        mass_max: Maximum halo mass for mass function [M_sun]
        use_mcmc_correction: Whether to apply MCMC S-dependent corrections
    """
    r_min: float = 1e-4  # Mpc
    r_max: float = 1.0   # Mpc
    n_bins: int = 50
    mass_min: float = 1e8   # M_sun
    mass_max: float = 1e15  # M_sun
    use_mcmc_correction: bool = True
