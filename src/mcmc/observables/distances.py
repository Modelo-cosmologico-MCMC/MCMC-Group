from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

C_LIGHT = 299792.458  # km/s


def _ensure_sorted(z: np.ndarray, H: np.ndarray):
    idx = np.argsort(z)
    return z[idx], H[idx]


def comoving_distance(z: np.ndarray, H_of_z: np.ndarray) -> np.ndarray:
    z, H = _ensure_sorted(np.asarray(z, float), np.asarray(H_of_z, float))
    integrand = C_LIGHT / H
    dc = cumulative_trapezoid(integrand, z, initial=0.0)
    return dc


def luminosity_distance(z: np.ndarray, H_of_z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    dc = comoving_distance(z, H_of_z)
    return (1.0 + z) * dc


def distance_modulus(z: np.ndarray, H_of_z: np.ndarray, M: float = -19.3) -> np.ndarray:
    """
    Mu(z)=5 log10(dL/Mpc)+25 + ajuste M (degeneracion absoluta).
    Para datasets demo el M se absorbe.
    """
    dL = luminosity_distance(z, H_of_z)  # Mpc si H esta en km/s/Mpc
    mu = 5.0 * np.log10(np.maximum(dL, 1e-30)) + 25.0
    return mu + (M + 19.3)  # offset controlado
