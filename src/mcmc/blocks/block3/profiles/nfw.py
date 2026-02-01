"""Navarro-Frenk-White (NFW) density profile.

The NFW profile is the standard dark matter halo profile from CDM simulations:
    rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]

MCMC extension: In the MCMC framework, the NFW profile is modified by
the stratified present, where S_local varies with tensorial density.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.ontology import S_0


@dataclass(frozen=True)
class NFWParams:
    """Parameters for NFW profile.

    Attributes:
        rho_s: Scale density [M_sun / Mpc^3]
        r_s: Scale radius [Mpc]
        r_vir: Virial radius [Mpc]
        S_halo: Entropic coordinate of halo
    """
    rho_s: float = 1e7
    r_s: float = 0.02
    r_vir: float = 0.2
    S_halo: float = S_0

    @property
    def concentration(self) -> float:
        """Halo concentration c = r_vir / r_s."""
        return self.r_vir / self.r_s


def nfw_density(r: float | np.ndarray, params: NFWParams) -> float | np.ndarray:
    """NFW density profile.

    rho(r) = rho_s / [(r/r_s) * (1 + r/r_s)^2]

    Args:
        r: Radius [Mpc]
        params: NFW parameters

    Returns:
        Density [M_sun / Mpc^3]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_s
    x = np.maximum(x, 1e-10)  # Avoid singularity at r=0
    rho = params.rho_s / (x * (1 + x) ** 2)
    return float(rho) if np.isscalar(r) else rho


def nfw_mass(r: float | np.ndarray, params: NFWParams) -> float | np.ndarray:
    """Enclosed mass for NFW profile.

    M(<r) = 4*pi*rho_s*r_s^3 * [ln(1+x) - x/(1+x)]
    where x = r/r_s

    Args:
        r: Radius [Mpc]
        params: NFW parameters

    Returns:
        Enclosed mass [M_sun]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_s
    x = np.maximum(x, 1e-10)
    prefactor = 4 * np.pi * params.rho_s * params.r_s ** 3
    mass = prefactor * (np.log(1 + x) - x / (1 + x))
    return float(mass) if np.isscalar(r) else mass


def nfw_velocity(r: float | np.ndarray, params: NFWParams) -> float | np.ndarray:
    """Circular velocity for NFW profile.

    v_c(r) = sqrt(G * M(<r) / r)

    Args:
        r: Radius [Mpc]
        params: NFW parameters

    Returns:
        Circular velocity [km/s]
    """
    r_arr = np.asarray(r)
    mass = nfw_mass(r_arr, params)
    r_safe = np.maximum(r_arr, 1e-10)

    # G in units: Mpc^3 / (M_sun * s^2) -> need conversion to km/s
    # G = 4.302e-6 kpc * (km/s)^2 / M_sun
    # G = 4.302e-9 Mpc * (km/s)^2 / M_sun
    G_units = 4.302e-9  # Mpc * (km/s)^2 / M_sun

    v_c = np.sqrt(G_units * mass / r_safe)
    return float(v_c) if np.isscalar(r) else v_c


def nfw_potential(r: float | np.ndarray, params: NFWParams) -> float | np.ndarray:
    """Gravitational potential for NFW profile.

    Phi(r) = -4*pi*G*rho_s*r_s^2 * ln(1+r/r_s) / (r/r_s)

    Args:
        r: Radius [Mpc]
        params: NFW parameters

    Returns:
        Potential [(km/s)^2]
    """
    r_arr = np.asarray(r)
    x = r_arr / params.r_s
    x = np.maximum(x, 1e-10)

    G_units = 4.302e-9  # Mpc * (km/s)^2 / M_sun
    prefactor = -4 * np.pi * G_units * params.rho_s * params.r_s ** 2

    phi = prefactor * np.log(1 + x) / x
    return float(phi) if np.isscalar(r) else phi


class NFWProfile:
    """NFW profile with MCMC extensions.

    Supports standard NFW calculations and MCMC modifications
    through the stratified present.
    """

    def __init__(self, params: NFWParams | None = None):
        """Initialize NFW profile.

        Args:
            params: Profile parameters
        """
        self.params = params or NFWParams()

    def density(self, r: float | np.ndarray) -> float | np.ndarray:
        """Density at radius r."""
        return nfw_density(r, self.params)

    def mass(self, r: float | np.ndarray) -> float | np.ndarray:
        """Enclosed mass at radius r."""
        return nfw_mass(r, self.params)

    def velocity(self, r: float | np.ndarray) -> float | np.ndarray:
        """Circular velocity at radius r."""
        return nfw_velocity(r, self.params)

    def potential(self, r: float | np.ndarray) -> float | np.ndarray:
        """Gravitational potential at radius r."""
        return nfw_potential(r, self.params)

    def density_mcmc(
        self,
        r: float | np.ndarray,
        S_local: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """MCMC-corrected density with stratified present.

        In the MCMC framework, local S affects the effective density:
            rho_eff = rho_NFW * f(S_local/S_global)

        For regions with S_local < S_global (tensorial islands),
        the effective dark matter density is enhanced.

        Args:
            r: Radius [Mpc]
            S_local: Local entropic coordinate (default: S_halo)

        Returns:
            MCMC-corrected density
        """
        rho_base = self.density(r)

        if S_local is None:
            return rho_base

        S_local_arr = np.asarray(S_local)
        S_global = self.params.S_halo

        # MCMC correction: denser regions have lower local S
        # This enhances effective dark matter in dense regions
        S_ratio = S_local_arr / S_global
        correction = 1.0 / np.maximum(S_ratio, 0.1)  # Cap enhancement

        return rho_base * correction

    @property
    def M_vir(self) -> float:
        """Virial mass."""
        return nfw_mass(self.params.r_vir, self.params)

    @property
    def V_max(self) -> float:
        """Maximum circular velocity."""
        # V_max occurs at r ~ 2.16 * r_s for NFW
        r_max = 2.16 * self.params.r_s
        return nfw_velocity(r_max, self.params)


def create_nfw_from_concentration(
    M_vir: float,
    c: float,
    z: float = 0.0,
    S_halo: float = S_0,
) -> NFWProfile:
    """Create NFW profile from virial mass and concentration.

    Args:
        M_vir: Virial mass [M_sun]
        c: Concentration
        z: Redshift (for Delta_vir calculation)
        S_halo: Entropic coordinate

    Returns:
        NFWProfile
    """
    # Delta_vir ~ 200 for z=0
    Delta_vir = 200.0

    # Critical density at z=0 [M_sun / Mpc^3]
    # rho_crit ~ 2.775e11 h^2 M_sun/Mpc^3
    h = 0.674
    rho_crit = 2.775e11 * h ** 2

    # Virial radius from M_vir
    r_vir = (3 * M_vir / (4 * np.pi * Delta_vir * rho_crit)) ** (1 / 3)

    # Scale radius
    r_s = r_vir / c

    # Scale density from NFW normalization
    # M_vir = 4*pi*rho_s*r_s^3 * f(c) where f(c) = ln(1+c) - c/(1+c)
    f_c = np.log(1 + c) - c / (1 + c)
    rho_s = M_vir / (4 * np.pi * r_s ** 3 * f_c)

    params = NFWParams(rho_s=rho_s, r_s=r_s, r_vir=r_vir, S_halo=S_halo)
    return NFWProfile(params)
