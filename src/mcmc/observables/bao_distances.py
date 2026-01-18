from __future__ import annotations

import numpy as np
from mcmc.observables.distances import comoving_distance, C_LIGHT


def angular_diameter_distance(z: np.ndarray, H_of_z: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de diametro angular D_A(z).

    D_A(z) = r(z) / (1+z)

    Donde r(z) es la distancia comoving.

    Args:
        z: Array de redshifts
        H_of_z: Array de H(z) correspondiente [km/s/Mpc]

    Returns:
        Array de D_A(z) [Mpc]
    """
    z = np.asarray(z, float)
    r = comoving_distance(z, H_of_z)
    return r / (1.0 + z)


def volume_distance(z: np.ndarray, H_of_z: np.ndarray) -> np.ndarray:
    """
    Calcula la distancia de volumen D_V(z) usada en BAO.

    D_V(z) = [ (1+z)^2 * D_A^2(z) * c*z / H(z) ]^(1/3)

    Esta es la distancia efectiva de volumen que combina
    la distancia angular y el factor de Hubble.

    Args:
        z: Array de redshifts
        H_of_z: Array de H(z) correspondiente [km/s/Mpc]

    Returns:
        Array de D_V(z) [Mpc]
    """
    z = np.asarray(z, float)
    DA = angular_diameter_distance(z, H_of_z)
    Hz = np.asarray(H_of_z, float)

    term = (1.0 + z) ** 2 * DA ** 2 * (C_LIGHT * z) / np.maximum(Hz, 1e-18)
    return np.maximum(term, 1e-30) ** (1.0 / 3.0)


def dv_over_rd(z: np.ndarray, H_of_z: np.ndarray, rd: float) -> np.ndarray:
    """
    Calcula D_V(z) / r_d - observable BAO tipico.

    Este es el observable que se compara con datos BAO
    (BOSS, eBOSS, etc.).

    Args:
        z: Array de redshifts
        H_of_z: Array de H(z) correspondiente [km/s/Mpc]
        rd: Sound horizon at drag epoch [Mpc]

    Returns:
        Array de D_V(z)/r_d (adimensional)

    Raises:
        ValueError: Si rd <= 0
    """
    rd = float(rd)
    if rd <= 0:
        raise ValueError("rd debe ser > 0")
    return volume_distance(z, H_of_z) / rd


def da_over_rd(z: np.ndarray, H_of_z: np.ndarray, rd: float) -> np.ndarray:
    """
    Calcula D_A(z) / r_d - observable BAO alternativo.

    Args:
        z: Array de redshifts
        H_of_z: Array de H(z) correspondiente [km/s/Mpc]
        rd: Sound horizon at drag epoch [Mpc]

    Returns:
        Array de D_A(z)/r_d (adimensional)

    Raises:
        ValueError: Si rd <= 0
    """
    rd = float(rd)
    if rd <= 0:
        raise ValueError("rd debe ser > 0")
    return angular_diameter_distance(z, H_of_z) / rd
