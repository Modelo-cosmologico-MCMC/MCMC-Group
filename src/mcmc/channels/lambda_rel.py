"""Constante Cosmológica Relativa Λ_rel(z).

La constante cosmológica relativa reemplaza Λ fija por una función
dinámica de z (y por tanto de S).

Ecuación canónica:
    Λ_rel(z) = Λ_0 [1 + ε(z_trans - z)],    z ≤ z_trans
    Λ_rel(z) = 8πG [ρ_id(z) + ρ_lat(z)]

Parámetros típicos:
    ε = 0.012 ± 0.003
    z_trans = 8.9 ± 0.4
    Δz = 1.5 (anchura transición)

Esta implementación generaliza ΛCDM para incluir las correcciones
tensionales del MCMC.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class LambdaRelParams:
    """Parámetros de Λ_rel(z).

    Attributes:
        Lambda_0: Valor de Λ normalizado a z=0 (en unidades de 3H₀²Ω_Λ)
        epsilon: Magnitud de la desviación
        z_trans: Redshift de transición
        dz: Anchura de la transición
        use_smooth: Si True, usa transición suave (tanh)
    """
    Lambda_0: float = 1.0
    epsilon: float = 0.012
    z_trans: float = 8.9
    dz: float = 1.5
    use_smooth: bool = True


def Lambda_rel_of_z(
    z: float | np.ndarray,
    params: LambdaRelParams
) -> float | np.ndarray:
    """Calcula Λ_rel(z) con transición suave.

    Λ_rel(z) = Λ_0 [1 + ε · (z_trans - z) · transition(z)]

    La función transition(z) es 1 para z << z_trans y 0 para z >> z_trans.

    Args:
        z: Redshift
        params: Parámetros de Λ_rel

    Returns:
        Λ_rel normalizado
    """
    z_arr = np.asarray(z)
    p = params

    if p.use_smooth:
        # Transición suave tipo tanh
        transition = 0.5 * (1.0 + np.tanh((p.z_trans - z_arr) / max(p.dz, 1e-6)))
    else:
        # Transición abrupta
        transition = np.where(z_arr <= p.z_trans, 1.0, 0.0)

    # Λ_rel(z) = Λ_0 [1 + ε · (z_trans - z) · transition]
    deviation = p.epsilon * (p.z_trans - z_arr) * transition

    return p.Lambda_0 * (1.0 + deviation)


def dLambda_rel_dz(
    z: float | np.ndarray,
    params: LambdaRelParams
) -> float | np.ndarray:
    """Derivada dΛ_rel/dz.

    Args:
        z: Redshift
        params: Parámetros de Λ_rel

    Returns:
        Derivada dΛ_rel/dz
    """
    z_arr = np.asarray(z)

    # Derivada numérica
    dz = 1e-5
    Lam_plus = Lambda_rel_of_z(z_arr + dz, params)
    Lam_minus = Lambda_rel_of_z(z_arr - dz, params)

    return (Lam_plus - Lam_minus) / (2 * dz)


def Omega_Lambda_rel(
    z: float | np.ndarray,
    params: LambdaRelParams,
    Omega_Lambda_0: float = 0.685
) -> float | np.ndarray:
    """Ω_Λ(z) efectivo con Λ_rel.

    Ω_Λ,rel(z) = Ω_Λ,0 · Λ_rel(z) / E²(z)

    donde E(z) = H(z)/H_0.

    Aproximación simple: Ω_Λ,rel(z) ≈ Ω_Λ,0 · Λ_rel(z) / Λ_0

    Args:
        z: Redshift
        params: Parámetros de Λ_rel
        Omega_Lambda_0: Ω_Λ actual

    Returns:
        Ω_Λ efectivo
    """
    Lam_rel = Lambda_rel_of_z(z, params)
    return Omega_Lambda_0 * Lam_rel / params.Lambda_0


def H_squared_correction(
    z: float | np.ndarray,
    params: LambdaRelParams,
    H0: float = 67.4,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685
) -> float | np.ndarray:
    """Corrección a H²(z) por Λ_rel.

    H²(z) = H₀² [Ω_m(1+z)³ + Ω_Λ · Λ_rel(z)/Λ_0]

    Args:
        z: Redshift
        params: Parámetros de Λ_rel
        H0: Constante de Hubble km/s/Mpc
        Omega_m: Fracción de materia
        Omega_Lambda: Fracción de energía oscura

    Returns:
        H²(z) en (km/s/Mpc)²
    """
    z_arr = np.asarray(z)

    # Término de materia
    matter_term = Omega_m * (1.0 + z_arr)**3

    # Término de Λ_rel
    Lam_rel = Lambda_rel_of_z(z, params)
    lambda_term = Omega_Lambda * Lam_rel / params.Lambda_0

    return H0**2 * (matter_term + lambda_term)


def H_rel(
    z: float | np.ndarray,
    params: LambdaRelParams,
    H0: float = 67.4,
    Omega_m: float = 0.315,
    Omega_Lambda: float = 0.685
) -> float | np.ndarray:
    """H(z) con Λ_rel.

    Args:
        z: Redshift
        params: Parámetros de Λ_rel
        H0: Constante de Hubble km/s/Mpc
        Omega_m: Fracción de materia
        Omega_Lambda: Fracción de energía oscura

    Returns:
        H(z) en km/s/Mpc
    """
    H2 = H_squared_correction(z, params, H0, Omega_m, Omega_Lambda)
    return np.sqrt(np.maximum(H2, 0.0))


def w_eff_Lambda(
    z: float | np.ndarray,
    params: LambdaRelParams
) -> float | np.ndarray:
    """Ecuación de estado efectiva w_eff de Λ_rel.

    Para una componente con densidad ρ que no escala como (1+z)^3(1+w):
    w_eff = -1 + (1/3) d ln ρ / d ln(1+z)

    Para Λ_rel:
    w_eff ≈ -1 + (1/3) (1+z)/Λ_rel · dΛ_rel/dz

    Args:
        z: Redshift
        params: Parámetros de Λ_rel

    Returns:
        Ecuación de estado efectiva
    """
    z_arr = np.asarray(z)

    Lam_rel = Lambda_rel_of_z(z, params)
    dLam_dz = dLambda_rel_dz(z, params)

    # w_eff = -1 + (1/3) · (1+z)/Λ_rel · dΛ_rel/dz
    Lam_safe = np.maximum(Lam_rel, 1e-30)
    w = -1.0 + (1.0 / 3.0) * (1.0 + z_arr) * dLam_dz / Lam_safe

    return w


@dataclass
class LambdaRelFromChannels:
    """Λ_rel construido desde canales ρ_id y ρ_lat.

    Λ_rel(z) = 8πG [ρ_id(z) + ρ_lat(z)]

    Attributes:
        G: Constante gravitatoria
        rho_id_func: Función ρ_id(z)
        rho_lat_func: Función ρ_lat(z)
    """
    G: float = 1.0
    rho_id_func: Callable[[np.ndarray], np.ndarray] | None = None
    rho_lat_func: Callable[[np.ndarray], np.ndarray] | None = None

    def Lambda_rel(self, z: float | np.ndarray) -> float | np.ndarray:
        """Calcula Λ_rel desde los canales.

        Args:
            z: Redshift

        Returns:
            Λ_rel(z)
        """
        z_arr = np.asarray(z)

        rho_id = self.rho_id_func(z_arr) if self.rho_id_func else np.zeros_like(z_arr)
        rho_lat = self.rho_lat_func(z_arr) if self.rho_lat_func else np.zeros_like(z_arr)

        return 8.0 * np.pi * self.G * (rho_id + rho_lat)
