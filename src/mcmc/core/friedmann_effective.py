from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from mcmc.channels.rho_id_refined import RhoIDRefinedParams, rho_id_refined


@dataclass(frozen=True)
class EffectiveParams:
    """
    Bloque II - Cierre efectivo para validacion BAO/H(z)/SNe.

    H(z) normalizado para que H(0) = H0:
      H(z) = H0 * sqrt( [rho_bar(z) + rho_id(z)] / [rho_bar(0) + rho_id(0)] )

    Donde:
      rho_bar(z) = rho_b0 * (1+z)^3  (coeficiente efectivo de materia)
      rho_id(z) = rho_id_refined(z; rho0, z_trans, eps)

    Esta normalizacion garantiza consistencia: H(z=0) = H0 exactamente.

    Attributes:
        H0: Constante de Hubble [km/s/Mpc]
        rho_b0: Coeficiente efectivo de materia (barionica/agrupada)
        rho_id: Parametros del canal indeterminado refinado
    """
    H0: float = 67.4
    rho_b0: float = 0.30
    rho_id: RhoIDRefinedParams = field(default_factory=RhoIDRefinedParams)


def rho_bar(z: np.ndarray, rho_b0: float) -> np.ndarray:
    """
    Calcula la densidad efectiva de materia.

    rho_bar(z) = rho_b0 * (1+z)^3

    Args:
        z: Array de redshifts
        rho_b0: Coeficiente de densidad de materia

    Returns:
        Array de rho_bar(z)
    """
    z = np.asarray(z, dtype=float)
    return np.maximum(rho_b0, 0.0) * (1.0 + z) ** 3


def rho_total(z: np.ndarray, p: EffectiveParams) -> np.ndarray:
    """
    Calcula la densidad total efectiva.

    rho_total(z) = rho_bar(z) + rho_id(z)

    Args:
        z: Array de redshifts
        p: Parametros efectivos

    Returns:
        Array de rho_total(z)
    """
    rb = rho_bar(z, p.rho_b0)
    rid = rho_id_refined(z, p.rho_id)
    return rb + rid


def H_of_z(z: np.ndarray, p: EffectiveParams) -> np.ndarray:
    """
    Calcula H(z) normalizado para que H(0) = H0.

    H(z) = H0 * sqrt( rho_total(z) / rho_total(0) )

    Esta forma garantiza H(z=0) = H0 exactamente.

    Args:
        z: Array de redshifts
        p: Parametros efectivos

    Returns:
        Array de H(z) [km/s/Mpc]
    """
    z = np.asarray(z, dtype=float)

    # Numerador: rho_total(z)
    rho_z = rho_total(z, p)

    # Denominador: rho_total(0)
    rho_0 = rho_total(np.array([0.0]), p)[0]
    denom = max(rho_0, 1e-18)

    # E^2(z) = rho(z) / rho(0)
    e2 = rho_z / denom

    return p.H0 * np.sqrt(np.maximum(e2, 1e-18))


def E_of_z(z: np.ndarray, p: EffectiveParams) -> np.ndarray:
    """
    Calcula E(z) = H(z)/H0.

    E(z) = sqrt( rho_total(z) / rho_total(0) )

    Args:
        z: Array de redshifts
        p: Parametros efectivos

    Returns:
        Array de E(z)
    """
    return H_of_z(z, p) / p.H0
