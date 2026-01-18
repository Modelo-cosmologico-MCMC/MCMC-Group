from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from mcmc.channels.rho_id_parametric import RhoIDParams, rho_id_of_z
from mcmc.channels.rho_lat_parametric import RhoLatParams, rho_lat_of_S


@dataclass(frozen=True)
class FriedmannParams:
    """
    Parametros del Friedmann modificado MCMC.

    Incluye los parametros cosmologicos estandar mas los
    canales oscuros rho_id y rho_lat.

    Attributes:
        H0: Constante de Hubble [km/s/Mpc]
        omega_m0: Densidad de materia hoy
        omega_r0: Densidad de radiacion hoy
        omega_k0: Curvatura
        rho_id: Parametros del canal indeterminado
        rho_lat: Parametros del canal latente
    """
    H0: float = 67.4
    omega_m0: float = 0.315
    omega_r0: float = 0.0
    omega_k0: float = 0.0
    rho_id: RhoIDParams = field(default_factory=RhoIDParams)
    rho_lat: RhoLatParams = field(default_factory=RhoLatParams)


def E2_of_z(z: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula E(z)^2 = (H/H0)^2 para el modelo LCDM base.

    Args:
        z: Array de redshifts
        p: Parametros de Friedmann

    Returns:
        Array de E^2 para cada z
    """
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)

    # Componentes estandar
    omega_de = 1.0 - p.omega_m0 - p.omega_r0 - p.omega_k0

    e2 = (
        p.omega_r0 * a ** -4
        + p.omega_m0 * a ** -3
        + p.omega_k0 * a ** -2
        + omega_de
    )

    return np.clip(e2, 1e-18, np.inf)


def E2_of_z_S(z: np.ndarray, S: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula E(z)^2 = (H/H0)^2 con correcciones rho_id(z) y rho_lat(S).

    Esta es la version modificada del Friedmann que incluye
    los canales oscuros del modelo MCMC.

    Args:
        z: Array de redshifts
        S: Array de entropia (mismo shape que z)
        p: Parametros de Friedmann

    Returns:
        Array de E^2 para cada (z, S)
    """
    z = np.asarray(z, dtype=float)
    S = np.asarray(S, dtype=float)

    # Base LCDM
    e2_base = E2_of_z(z, p)

    # Correcciones MCMC
    rid = rho_id_of_z(z, p.rho_id)
    rlat = rho_lat_of_S(S, p.rho_lat)

    # Acople aditivo (placeholder).
    # Sustituye por tu ecuacion definitiva con pesos Mp/Ep, PhiAd, etc.
    # Normalizamos por rho_crit para que sea adimensional
    rho_crit_factor = 1.0  # Ajustar segun unidades

    e2_corrected = e2_base + rho_crit_factor * (rid + rlat)

    return np.clip(e2_corrected, 1e-18, np.inf)


def H_of_z(z: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula H(z) para el modelo LCDM base.

    Args:
        z: Array de redshifts
        p: Parametros de Friedmann

    Returns:
        Array de H(z) [km/s/Mpc]
    """
    return p.H0 * np.sqrt(E2_of_z(z, p))


def H_of_z_S(z: np.ndarray, S: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula H(z) con correcciones MCMC.

    Args:
        z: Array de redshifts
        S: Array de entropia
        p: Parametros de Friedmann

    Returns:
        Array de H(z) [km/s/Mpc]
    """
    return p.H0 * np.sqrt(E2_of_z_S(z, S, p))


def w_eff_of_z_S(z: np.ndarray, S: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula la ecuacion de estado efectiva w_eff(z).

    w_eff = -1 - (2/3) * (1/E^2) * dE^2/d(ln a)

    Args:
        z: Array de redshifts
        S: Array de entropia
        p: Parametros de Friedmann

    Returns:
        Array de w_eff para cada (z, S)
    """
    z = np.asarray(z, dtype=float)
    e2 = E2_of_z_S(z, S, p)

    # Derivada numerica respecto a ln(a) = -ln(1+z)
    dz = 1e-4
    z_plus = z + dz
    z_minus = z - dz
    z_minus = np.maximum(z_minus, 0.0)

    e2_plus = E2_of_z_S(z_plus, S, p)
    e2_minus = E2_of_z_S(z_minus, S, p)

    # d(ln a) = -dz/(1+z)
    dlna = -dz / (1.0 + z)
    de2_dlna = (e2_plus - e2_minus) / (2.0 * dlna)

    w_eff = -1.0 - (2.0 / 3.0) * de2_dlna / np.maximum(e2, 1e-18)

    return w_eff


def q_of_z_S(z: np.ndarray, S: np.ndarray, p: FriedmannParams) -> np.ndarray:
    """
    Calcula el parametro de deceleracion q(z).

    q = -1 - (d ln H / d ln a)

    Args:
        z: Array de redshifts
        S: Array de entropia
        p: Parametros de Friedmann

    Returns:
        Array de q para cada (z, S)
    """
    z = np.asarray(z, dtype=float)

    # Derivada numerica
    dz = 1e-4
    z_plus = z + dz
    z_minus = np.maximum(z - dz, 0.0)

    H_plus = H_of_z_S(z_plus, S, p)
    H_minus = H_of_z_S(z_minus, S, p)

    dlnH = (np.log(H_plus) - np.log(H_minus)) / (2.0 * dz)
    dlna_dz = -1.0 / (1.0 + z)

    dlnH_dlna = dlnH / dlna_dz

    return -1.0 - dlnH_dlna
