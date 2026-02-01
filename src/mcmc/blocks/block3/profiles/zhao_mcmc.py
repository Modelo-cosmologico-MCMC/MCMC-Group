"""Zhao-MCMC density profile.

The Zhao profile is a generalized double power-law:
    rho(r) = rho_s / [(r/r_s)^gamma * (1 + (r/r_s)^alpha)^((beta-gamma)/alpha)]

MCMC extension: The inner slope gamma depends on local S:
    gamma(S) = gamma_0 + delta_gamma * (S_GEOM / S_local)

This produces:
- Cuspy profiles (gamma ~ 1) for high S_local (low density)
- Cored profiles (gamma ~ 0) for low S_local (tensorial islands)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

from mcmc.core.ontology import S_0, S_GEOM


@dataclass(frozen=True)
class ZhaoMCMCParams:
    """Parameters for Zhao-MCMC profile.

    Attributes:
        rho_s: Scale density [M_sun / Mpc^3]
        r_s: Scale radius [Mpc]
        alpha: Transition sharpness (typically ~1)
        beta: Outer slope (typically ~3)
        gamma_0: Base inner slope (typically ~1 for NFW-like)
        delta_gamma: S-dependent correction strength
        S_halo: Global entropic coordinate of halo
    """
    rho_s: float = 1e7
    r_s: float = 0.02
    alpha: float = 1.0
    beta: float = 3.0
    gamma_0: float = 1.0
    delta_gamma: float = 0.5
    S_halo: float = S_0


def zhao_gamma(S: float | np.ndarray, params: ZhaoMCMCParams) -> float | np.ndarray:
    """Compute S-dependent inner slope gamma(S).

    gamma(S) = gamma_0 - delta_gamma * (S_GEOM / S)

    For large S (~ S_0): gamma ~ gamma_0 (NFW-like cusp)
    For small S (~ S_GEOM): gamma ~ gamma_0 - delta_gamma (cored)

    Args:
        S: Entropic coordinate
        params: Profile parameters

    Returns:
        Inner slope gamma
    """
    S_arr = np.asarray(S)
    S_safe = np.maximum(S_arr, S_GEOM)

    # gamma decreases for smaller S (more cored in dense regions)
    gamma = params.gamma_0 - params.delta_gamma * (S_GEOM / S_safe)
    gamma = np.clip(gamma, 0.0, 2.0)  # Physical bounds

    return float(gamma) if np.isscalar(S) else gamma


def zhao_mcmc_density(
    r: float | np.ndarray,
    params: ZhaoMCMCParams,
    S_local: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Zhao-MCMC density profile.

    rho(r) = rho_s / [(r/r_s)^gamma * (1 + (r/r_s)^alpha)^((beta-gamma)/alpha)]

    where gamma = gamma(S_local) depends on local entropic coordinate.

    Args:
        r: Radius [Mpc]
        params: Profile parameters
        S_local: Local entropic coordinate (default: S_halo)

    Returns:
        Density [M_sun / Mpc^3]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_s
    x = np.maximum(x, 1e-10)

    # Determine gamma from local S
    if S_local is None:
        S_local = params.S_halo
    gamma = zhao_gamma(S_local, params)

    # Generalized Zhao profile
    exponent = (params.beta - gamma) / params.alpha
    rho = params.rho_s / (x ** gamma * (1 + x ** params.alpha) ** exponent)

    return float(rho) if np.isscalar(r) and np.isscalar(S_local) else rho


def zhao_mcmc_mass(
    r: float | np.ndarray,
    params: ZhaoMCMCParams,
    S_local: float | None = None,
) -> float | np.ndarray:
    """Enclosed mass for Zhao-MCMC profile (numerical integration).

    M(<r) = 4*pi * integral_0^r rho(r') * r'^2 dr'

    Args:
        r: Radius [Mpc]
        params: Profile parameters
        S_local: Local entropic coordinate

    Returns:
        Enclosed mass [M_sun]
    """
    r_arr = np.atleast_1d(r)
    masses = np.zeros_like(r_arr)

    for i, r_val in enumerate(r_arr):
        def integrand(r_prime):
            rho = zhao_mcmc_density(r_prime, params, S_local)
            return 4 * np.pi * rho * r_prime ** 2

        mass, _ = quad(integrand, 1e-10 * params.r_s, r_val)
        masses[i] = mass

    return float(masses[0]) if np.isscalar(r) else masses


def zhao_mcmc_velocity(
    r: float | np.ndarray,
    params: ZhaoMCMCParams,
    S_local: float | None = None,
) -> float | np.ndarray:
    """Circular velocity for Zhao-MCMC profile.

    v_c(r) = sqrt(G * M(<r) / r)

    Args:
        r: Radius [Mpc]
        params: Profile parameters
        S_local: Local entropic coordinate

    Returns:
        Circular velocity [km/s]
    """
    r_arr = np.asarray(r)
    mass = zhao_mcmc_mass(r_arr, params, S_local)
    r_safe = np.maximum(r_arr, 1e-10)

    G_units = 4.302e-9  # Mpc * (km/s)^2 / M_sun
    v_c = np.sqrt(G_units * mass / r_safe)

    return float(v_c) if np.isscalar(r) else v_c


class ZhaoMCMCProfile:
    """Zhao-MCMC profile with S-dependent inner slope.

    This profile interpolates between:
    - NFW-like cuspy profile for regions with S ~ S_0
    - Burkert-like cored profile for tensorial islands (S << S_0)

    The transition is controlled by the stratified present.
    """

    def __init__(self, params: ZhaoMCMCParams | None = None):
        """Initialize Zhao-MCMC profile.

        Args:
            params: Profile parameters
        """
        self.params = params or ZhaoMCMCParams()

    def gamma(self, S: float | np.ndarray | None = None) -> float | np.ndarray:
        """Inner slope at given S."""
        if S is None:
            S = self.params.S_halo
        return zhao_gamma(S, self.params)

    def density(
        self,
        r: float | np.ndarray,
        S_local: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Density at radius r."""
        return zhao_mcmc_density(r, self.params, S_local)

    def mass(
        self,
        r: float | np.ndarray,
        S_local: float | None = None,
    ) -> float | np.ndarray:
        """Enclosed mass at radius r."""
        return zhao_mcmc_mass(r, self.params, S_local)

    def velocity(
        self,
        r: float | np.ndarray,
        S_local: float | None = None,
    ) -> float | np.ndarray:
        """Circular velocity at radius r."""
        return zhao_mcmc_velocity(r, self.params, S_local)

    def is_cored(self, S_local: float | None = None) -> bool:
        """Check if profile is cored at given S.

        Profile is considered cored if gamma < 0.5.

        Args:
            S_local: Local entropic coordinate

        Returns:
            True if profile is cored
        """
        gamma = self.gamma(S_local)
        return bool(gamma < 0.5)

    def is_cuspy(self, S_local: float | None = None) -> bool:
        """Check if profile is cuspy at given S.

        Profile is considered cuspy if gamma > 0.5.

        Args:
            S_local: Local entropic coordinate

        Returns:
            True if profile is cuspy
        """
        return not self.is_cored(S_local)

    def profile_type(self, S_local: float | None = None) -> str:
        """Get profile type description.

        Args:
            S_local: Local entropic coordinate

        Returns:
            "cored", "cuspy", or "intermediate"
        """
        gamma = self.gamma(S_local)
        if gamma < 0.3:
            return "cored"
        elif gamma > 0.7:
            return "cuspy"
        else:
            return "intermediate"


def create_zhao_mcmc_interpolated(
    M_vir: float,
    c: float,
    S_halo: float = S_0,
    core_strength: float = 0.5,
) -> ZhaoMCMCProfile:
    """Create Zhao-MCMC profile that interpolates NFW/Burkert.

    Args:
        M_vir: Virial mass [M_sun]
        c: Concentration
        S_halo: Global entropic coordinate
        core_strength: delta_gamma parameter (0 = pure NFW, 1 = strong coring)

    Returns:
        ZhaoMCMCProfile
    """
    # Use NFW-like outer parameters
    h = 0.674
    rho_crit = 2.775e11 * h ** 2
    Delta_vir = 200.0

    r_vir = (3 * M_vir / (4 * np.pi * Delta_vir * rho_crit)) ** (1 / 3)
    r_s = r_vir / c

    f_c = np.log(1 + c) - c / (1 + c)
    rho_s = M_vir / (4 * np.pi * r_s ** 3 * f_c)

    params = ZhaoMCMCParams(
        rho_s=rho_s,
        r_s=r_s,
        alpha=1.0,
        beta=3.0,
        gamma_0=1.0,
        delta_gamma=core_strength,
        S_halo=S_halo,
    )
    return ZhaoMCMCProfile(params)
