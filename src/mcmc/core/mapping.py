"""Cosmological mappings: z↔t and z↔S for post-Big-Bang regime.

These functions map between redshift z, cosmic time t, and entropic S
in the observable universe (post-BB, S > S_BB).

Convention:
    - z = 0 is TODAY (present)
    - z > 0 is the past (higher z = earlier time)
    - t = 0 is the Big Bang (S = S_BB)
    - t > 0 is cosmic time since Big Bang
    - t_lookback(z) is lookback time from today to redshift z

Units:
    - H(z) in km/s/Mpc
    - Time outputs in Gyr (when using compute_universe_age_gyr)
    - Hubble time: t_H = 1/H0 ≈ 977.8/H0 Gyr (for H0 in km/s/Mpc)
"""
from __future__ import annotations

from typing import Callable
import numpy as np
from scipy.interpolate import interp1d

from mcmc.core.chronos import ChronosParams, S_of_t


# Physical constants for unit conversion
# Conversion factor: integral of dz/[(1+z)H(z)] with H in km/s/Mpc gives time in Gyr
# when multiplied by this factor.
# Derivation: 1 Mpc = 3.085677581e19 km, 1 Gyr = 3.15576e16 s
# Factor = (km/Mpc) * (s/Gyr) = 3.085677581e19 / 3.15576e16 ≈ 977.79 Gyr * (km/s/Mpc)
KM_PER_MPC = 3.085677581e19  # km per Mpc
SEC_PER_GYR = 3.15576e16     # seconds per Gyr
HUBBLE_TIME_FACTOR = KM_PER_MPC / SEC_PER_GYR  # ≈ 977.79 Gyr * km/s/Mpc


def make_interpolators(z: np.ndarray, y: np.ndarray):
    """Create y(z) interpolator. Assumes z monotonically increasing."""
    if not np.all(np.diff(z) > 0):
        idx = np.argsort(z)
        z = z[idx]
        y = y[idx]
    return interp1d(z, y, kind="cubic", bounds_error=False, fill_value="extrapolate")


def S_to_z(a: np.ndarray) -> np.ndarray:
    """Convert scale factor to redshift: z = 1/a - 1."""
    return 1.0 / np.asarray(a) - 1.0


def z_to_a(z: np.ndarray) -> np.ndarray:
    """Convert redshift to scale factor: a = 1/(1+z)."""
    return 1.0 / (1.0 + np.asarray(z))


def t_lookback_of_z(
    z: np.ndarray,
    H_of_z: Callable[[np.ndarray], np.ndarray],
    c_over_H0: float = 1.0,
) -> np.ndarray:
    """Compute lookback time from today (z=0) to redshift z.

    t_lookback(z) = integral_0^z dz' / [(1+z') H(z')]

    This is the time elapsed between z and today.
    Units depend on c_over_H0 and H(z) units.

    Args:
        z: Redshift array
        H_of_z: Function returning H(z) (same units as c_over_H0)
        c_over_H0: Speed of light / H0 for unit conversion (default: 1.0 = relative)

    Returns:
        Lookback time array (same shape as z)
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))

    # Sort for integration
    idx_sort = np.argsort(z_arr)
    z_sorted = z_arr[idx_sort]

    # Evaluate H(z)
    Hz = H_of_z(z_sorted)
    Hz = np.maximum(Hz, 1e-30)  # Avoid division by zero

    # Integrand: 1 / [(1+z) H(z)]
    integrand = c_over_H0 / ((1.0 + z_sorted) * Hz)

    # Cumulative trapezoid integration
    t_cumulative = np.zeros_like(z_sorted)
    for i in range(1, len(z_sorted)):
        dz = z_sorted[i] - z_sorted[i - 1]
        t_cumulative[i] = t_cumulative[i - 1] + 0.5 * dz * (integrand[i] + integrand[i - 1])

    # Map back to original order
    out = np.empty_like(z_arr)
    out[idx_sort] = t_cumulative
    return out


def t_cosmic_of_z(
    z: np.ndarray,
    H_of_z: Callable[[np.ndarray], np.ndarray],
    t0: float,
    c_over_H0: float = 1.0,
) -> np.ndarray:
    """Compute cosmic time since Big Bang for redshift z.

    t_cosmic(z) = t0 - t_lookback(z)

    where t0 is the age of the universe (time since BB to today).

    Args:
        z: Redshift array
        H_of_z: Function returning H(z)
        t0: Age of universe (time from BB to z=0)
        c_over_H0: Speed of light / H0 for unit conversion

    Returns:
        Cosmic time array (t=0 at Big Bang)
    """
    t_lookback = t_lookback_of_z(z, H_of_z, c_over_H0)
    return t0 - t_lookback


def z_of_t_lookback(
    t_lb: np.ndarray,
    H_of_z: Callable[[np.ndarray], np.ndarray],
    zmax: float = 20.0,
    n_grid: int = 500,
    c_over_H0: float = 1.0,
) -> np.ndarray:
    """Invert lookback time to get redshift.

    Given lookback time, returns the corresponding redshift.
    Uses interpolation on a pre-computed grid.

    Args:
        t_lb: Lookback time array
        H_of_z: Function returning H(z)
        zmax: Maximum z for grid
        n_grid: Grid resolution
        c_over_H0: Speed of light / H0 for unit conversion

    Returns:
        Redshift array
    """
    t_lb_arr = np.atleast_1d(np.asarray(t_lb, dtype=float))

    # Build z grid and compute lookback times
    z_grid = np.linspace(0, zmax, n_grid)
    t_grid = t_lookback_of_z(z_grid, H_of_z, c_over_H0)

    # Interpolate: t_lookback -> z
    interp = interp1d(t_grid, z_grid, kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp(t_lb_arr)


def S_of_z(
    z: np.ndarray,
    H_of_z: Callable[[np.ndarray], np.ndarray],
    chronos: ChronosParams,
    t0: float,
    c_over_H0: float = 1.0,
) -> np.ndarray:
    """Map redshift z to entropic S via z → t_cosmic → S.

    This composes the cosmological time mapping with Chronos inversion.

    Args:
        z: Redshift array
        H_of_z: Function returning H(z)
        chronos: Chronos parameters for S(t) inversion
        t0: Age of universe (for t_cosmic calculation)
        c_over_H0: Speed of light / H0 for unit conversion

    Returns:
        Entropic S array (S > S_BB for z >= 0)
    """
    t_cosmic = t_cosmic_of_z(z, H_of_z, t0, c_over_H0)
    return S_of_t(t_cosmic, chronos)


def compute_universe_age(
    H_of_z: Callable[[np.ndarray], np.ndarray],
    zmax: float = 1000.0,
    n_grid: int = 2000,
    in_gyr: bool = True,
) -> float:
    """Estimate the age of the universe by integrating to high z.

    t0 ≈ integral_0^zmax dz' / [(1+z') H(z')]

    Args:
        H_of_z: Function returning H(z) in km/s/Mpc
        zmax: Maximum z for integration (should be large)
        n_grid: Grid resolution
        in_gyr: If True (default), return age in Gyr. Otherwise dimensionless.

    Returns:
        Estimated age of universe in Gyr (if in_gyr=True) or dimensionless.

    Note:
        For ΛCDM with H0≈67.4 km/s/Mpc, typical age is ~13.8 Gyr.
    """
    z_grid = np.linspace(0, zmax, n_grid)

    # Use HUBBLE_TIME_FACTOR for Gyr conversion, or 1.0 for dimensionless
    factor = HUBBLE_TIME_FACTOR if in_gyr else 1.0

    t_grid = t_lookback_of_z(z_grid, H_of_z, c_over_H0=factor)
    return float(t_grid[-1])


def compute_hubble_time_gyr(H0: float) -> float:
    """Compute Hubble time t_H = 1/H0 in Gyr.

    Args:
        H0: Hubble constant in km/s/Mpc

    Returns:
        Hubble time in Gyr: t_H ≈ 977.79/H0 Gyr
    """
    return HUBBLE_TIME_FACTOR / H0
