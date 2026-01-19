"""Post-Big-Bang solver: Observable cosmology regime S > S_BB.

This solver handles the observable universe where:
- Cosmological observables: H(z), μ(z), BAO
- Friedmann dynamics with effective dark energy
- Redshift z ∈ [0, ∞) maps to cosmic time t > 0

CRITICAL: This solver operates ONLY in the post-BB regime (S > S_BB = 1.001).
Observables like H(z), SNe Ia distances, and BAO are defined here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mcmc.core.friedmann_effective import EffectiveParams, H_of_z
from mcmc.core.mapping import t_lookback_of_z, t_cosmic_of_z, compute_universe_age
from mcmc.channels.rho_id_refined import RhoIDRefinedParams
from mcmc.observables.distances import distance_modulus
from mcmc.observables.bao_distances import dv_over_rd


@dataclass
class PostBBResult:
    """Result from post-BB solver."""
    # Callable observables (functions of z)
    H_of_z: Callable[[np.ndarray], np.ndarray]
    mu_of_z: Callable[[np.ndarray], np.ndarray]
    DVrd_of_z: Callable[[np.ndarray], np.ndarray]

    # Cosmological parameters
    H0: float
    t0: float  # Age of universe (time since BB)

    # Calibration
    rd: float  # Sound horizon (BAO)
    M: float   # Supernova absolute magnitude

    # Effective dark energy parameters
    rho_id_params: RhoIDRefinedParams

    # Metadata
    params: dict = field(default_factory=dict)


@dataclass
class PostBBParams:
    """Parameters for post-BB solver."""
    # Hubble constant
    H0: float = 67.4  # km/s/Mpc

    # Matter density (baryonic + dark matter)
    rho_b0: float = 0.30

    # Effective dark energy (rho_id)
    rho0: float = 0.70
    z_trans: float = 1.0
    eps: float = 0.05

    # Calibration
    rd: float = 147.0  # Mpc
    M: float = -19.3   # mag

    # Integration settings
    zmax_age: float = 1000.0
    n_grid_age: int = 2000


def solve_postbb(params: PostBBParams | None = None) -> PostBBResult:
    """Solve the post-Big-Bang cosmology.

    Constructs observable functions H(z), μ(z), DV/rd(z) for the
    post-BB regime using effective Friedmann dynamics.

    Args:
        params: Post-BB parameters. Uses defaults if None.

    Returns:
        PostBBResult with callable observables and cosmological parameters.
    """
    if params is None:
        params = PostBBParams()

    # Build effective dark energy
    rho_id = RhoIDRefinedParams(
        rho0=params.rho0,
        z_trans=params.z_trans,
        eps=params.eps,
    )

    # Build Friedmann parameters
    eff_params = EffectiveParams(
        H0=params.H0,
        rho_b0=params.rho_b0,
        rho_id=rho_id,
    )

    # Create H(z) function
    def H_func(z: np.ndarray) -> np.ndarray:
        return H_of_z(np.asarray(z, float), eff_params)

    # Compute age of universe
    t0 = compute_universe_age(
        H_func,
        zmax=params.zmax_age,
        n_grid=params.n_grid_age,
    )

    # Create μ(z) function
    def mu_func(z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, float)
        return distance_modulus(z_arr, H_func(z_arr), M=params.M)

    # Create DV/rd(z) function
    def dvrd_func(z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, float)
        return dv_over_rd(z_arr, H_func(z_arr), rd=params.rd)

    return PostBBResult(
        H_of_z=H_func,
        mu_of_z=mu_func,
        DVrd_of_z=dvrd_func,
        H0=params.H0,
        t0=t0,
        rd=params.rd,
        M=params.M,
        rho_id_params=rho_id,
        params={
            "H0": params.H0,
            "rho_b0": params.rho_b0,
            "rho0": params.rho0,
            "z_trans": params.z_trans,
            "eps": params.eps,
            "rd": params.rd,
            "M": params.M,
        },
    )


def evaluate_postbb_at_z(result: PostBBResult, z: np.ndarray) -> dict:
    """Evaluate all observables at given redshifts.

    Args:
        result: PostBBResult from solve_postbb
        z: Redshift array

    Returns:
        Dictionary with H, mu, DVrd arrays at given z values.
    """
    z_arr = np.asarray(z, float)
    return {
        "z": z_arr,
        "H": result.H_of_z(z_arr),
        "mu": result.mu_of_z(z_arr),
        "DVrd": result.DVrd_of_z(z_arr),
    }


def get_postbb_time_mapping(
    result: PostBBResult,
    z: np.ndarray,
) -> dict:
    """Get time mappings for given redshifts.

    Args:
        result: PostBBResult from solve_postbb
        z: Redshift array

    Returns:
        Dictionary with:
        - t_lookback: Lookback time from today (z=0) to each z
        - t_cosmic: Cosmic time since Big Bang for each z
    """
    z_arr = np.asarray(z, float)
    t_lb = t_lookback_of_z(z_arr, result.H_of_z)
    t_cosmic = t_cosmic_of_z(z_arr, result.H_of_z, result.t0)

    return {
        "z": z_arr,
        "t_lookback": t_lb,
        "t_cosmic": t_cosmic,
    }
