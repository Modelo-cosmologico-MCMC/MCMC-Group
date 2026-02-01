"""Burkert density profile (cored profile).

The Burkert profile has a central core, unlike NFW:
    rho(r) = rho_0 / [(1 + r/r_0) * (1 + (r/r_0)^2)]

This profile better matches rotation curves of dwarf galaxies,
which is a key test for the MCMC model's MCV predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import S_0


@dataclass(frozen=True)
class BurkertParams:
    """Parameters for Burkert profile.

    Attributes:
        rho_0: Central density [M_sun / Mpc^3]
        r_0: Core radius [Mpc]
        r_max: Maximum radius for integration [Mpc]
        S_halo: Entropic coordinate of halo
    """
    rho_0: float = 1e8
    r_0: float = 0.005
    r_max: float = 0.1
    S_halo: float = S_0


def burkert_density(r: float | np.ndarray, params: BurkertParams) -> float | np.ndarray:
    """Burkert density profile.

    rho(r) = rho_0 / [(1 + r/r_0) * (1 + (r/r_0)^2)]

    Args:
        r: Radius [Mpc]
        params: Burkert parameters

    Returns:
        Density [M_sun / Mpc^3]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_0
    rho = params.rho_0 / ((1 + x) * (1 + x ** 2))
    return float(rho) if np.isscalar(r) else rho


def burkert_mass(r: float | np.ndarray, params: BurkertParams) -> float | np.ndarray:
    """Enclosed mass for Burkert profile.

    M(<r) = pi*rho_0*r_0^3 * [ln(1+x^2) + 2*ln(1+x) - 2*arctan(x)]
    where x = r/r_0

    Args:
        r: Radius [Mpc]
        params: Burkert parameters

    Returns:
        Enclosed mass [M_sun]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_0
    prefactor = np.pi * params.rho_0 * params.r_0 ** 3
    mass = prefactor * (np.log(1 + x ** 2) + 2 * np.log(1 + x) - 2 * np.arctan(x))
    return float(mass) if np.isscalar(r) else mass


def burkert_velocity(r: float | np.ndarray, params: BurkertParams) -> float | np.ndarray:
    """Circular velocity for Burkert profile.

    v_c(r) = sqrt(G * M(<r) / r)

    Args:
        r: Radius [Mpc]
        params: Burkert parameters

    Returns:
        Circular velocity [km/s]
    """
    r_arr = np.asarray(r)
    mass = burkert_mass(r_arr, params)
    r_safe = np.maximum(r_arr, 1e-10)

    G_units = 4.302e-9  # Mpc * (km/s)^2 / M_sun
    v_c = np.sqrt(G_units * mass / r_safe)
    return float(v_c) if np.isscalar(r) else v_c


class BurkertProfile:
    """Burkert profile with MCMC extensions.

    The cored Burkert profile is particularly relevant for MCMC
    because the MCV (dark matter equivalent) naturally produces
    cores in regions of high tensorial density.
    """

    def __init__(self, params: BurkertParams | None = None):
        """Initialize Burkert profile.

        Args:
            params: Profile parameters
        """
        self.params = params or BurkertParams()

    def density(self, r: float | np.ndarray) -> float | np.ndarray:
        """Density at radius r."""
        return burkert_density(r, self.params)

    def mass(self, r: float | np.ndarray) -> float | np.ndarray:
        """Enclosed mass at radius r."""
        return burkert_mass(r, self.params)

    def velocity(self, r: float | np.ndarray) -> float | np.ndarray:
        """Circular velocity at radius r."""
        return burkert_velocity(r, self.params)

    def density_mcmc(
        self,
        r: float | np.ndarray,
        S_local: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """MCMC-corrected density with stratified present.

        In MCMC, the core radius r_0 depends on local S:
            r_0_eff = r_0 * (S_global / S_local)^alpha

        For lower S_local (denser regions), the core is smaller.

        Args:
            r: Radius [Mpc]
            S_local: Local entropic coordinate

        Returns:
            MCMC-corrected density
        """
        if S_local is None:
            return self.density(r)

        S_local_arr = np.asarray(S_local)
        S_global = self.params.S_halo
        alpha = 0.5  # Core-S coupling strength

        # Effective core radius
        r_0_eff = self.params.r_0 * (S_global / np.maximum(S_local_arr, 1.0)) ** alpha

        # Compute density with effective core
        r_arr = np.asarray(r)
        x = r_arr / r_0_eff
        rho = self.params.rho_0 / ((1 + x) * (1 + x ** 2))

        return float(rho) if np.isscalar(r) else rho

    @property
    def M_core(self) -> float:
        """Mass within core radius."""
        return burkert_mass(self.params.r_0, self.params)

    @property
    def rho_central(self) -> float:
        """Central density."""
        return self.params.rho_0


def create_burkert_from_rotation(
    V_flat: float,
    r_flat: float,
    S_halo: float = S_0,
) -> BurkertProfile:
    """Create Burkert profile from rotation curve parameters.

    For a Burkert profile, the flat rotation velocity is related to:
        V_flat^2 ~ G * M(<r_flat) / r_flat

    Args:
        V_flat: Flat rotation velocity [km/s]
        r_flat: Radius where rotation curve flattens [Mpc]
        S_halo: Entropic coordinate

    Returns:
        BurkertProfile
    """
    G_units = 4.302e-9  # Mpc * (km/s)^2 / M_sun

    # Estimate mass at r_flat
    M_flat = V_flat ** 2 * r_flat / G_units

    # For Burkert, core radius ~ 0.5 * r_flat typically
    r_0 = 0.5 * r_flat

    # Solve for rho_0 from M(<r_flat)
    x = r_flat / r_0
    f_x = np.log(1 + x ** 2) + 2 * np.log(1 + x) - 2 * np.arctan(x)
    rho_0 = M_flat / (np.pi * r_0 ** 3 * f_x)

    params = BurkertParams(rho_0=rho_0, r_0=r_0, r_max=5 * r_flat, S_halo=S_halo)
    return BurkertProfile(params)
