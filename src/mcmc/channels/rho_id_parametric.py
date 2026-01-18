"""
Canal Indeterminado ρ_id - Modelo Paramétrico (Nivel A)
=======================================================

Este módulo implementa el canal de energía indeterminada ρ_id usando
una parametrización directa en z, ideal para:
- Validación rápida
- Ajuste con emcee
- Comparación con ΛCDM

La parametrización presenta una transición en z_trans:
    ρ_id(z) = ρ_0 * (1+z)^3           para z > z_trans
    ρ_id(z) = ρ_0 * [1 + ε*(z_trans - z)]   para z ≤ z_trans

Esto genera un "plateau" en épocas tardías, simulando la transición
de materia efectiva a comportamiento tipo constante cosmológica.

Parámetros del modelo refinado (MVP):
- H0: Constante de Hubble
- rho_id0: Densidad de ρ_id hoy (o Omega_id0)
- gamma: Factor de dilución
- z_trans: Redshift de transición
- epsilon: Parámetro de pendiente post-transición

Referencias:
    - Documento de Simulaciones Observacionales
    - MCMC Maestro: Nivel A (paramétrico/refinado)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline


# Constantes físicas
C_LIGHT = 299792.458  # km/s
H0_FIDUCIAL = 67.4    # km/s/Mpc
RHO_CRIT_OVER_H2 = 3.0 / (8.0 * np.pi)  # ρ_crit / (H0^2) en unidades G=1


@dataclass
class RhoIdParametricParams:
    """
    Parámetros del modelo paramétrico de ρ_id.

    Attributes:
        Omega_id0: Fracción de densidad de ρ_id hoy
        z_trans: Redshift de transición
        epsilon: Pendiente del plateau post-transición
        delta_z_trans: Suavizado de la transición (ancho tanh)
        gamma: Factor de dilución (para variante con decay)
    """
    Omega_id0: float = 0.7      # Fracción hoy (análogo a Omega_Lambda)
    z_trans: float = 0.5        # Redshift de transición
    epsilon: float = 0.01       # Pendiente del plateau
    delta_z_trans: float = 0.1  # Suavizado de la transición
    gamma: float = 0.0          # Factor de dilución (0 = sin decay)

    def to_dict(self) -> Dict[str, float]:
        """Convierte a diccionario."""
        return {
            'Omega_id0': self.Omega_id0,
            'z_trans': self.z_trans,
            'epsilon': self.epsilon,
            'delta_z_trans': self.delta_z_trans,
            'gamma': self.gamma,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'RhoIdParametricParams':
        """Construye desde diccionario."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def rho_id_sharp_transition(
    z: np.ndarray,
    params: RhoIdParametricParams,
    Omega_m0: float = 0.3
) -> np.ndarray:
    """
    Calcula ρ_id(z) con transición abrupta (piecewise).

    ρ_id(z) = ρ_0 * (1+z)^3           para z > z_trans
    ρ_id(z) = ρ_0 * [1 + ε*(z_trans - z)]   para z ≤ z_trans

    Args:
        z: Array de redshifts
        params: Parámetros del modelo
        Omega_m0: Fracción de materia (para normalización)

    Returns:
        Array con ρ_id/ρ_crit0 (en unidades de densidad crítica hoy)
    """
    z = np.atleast_1d(z)
    rho_id = np.zeros_like(z)

    # Normalización: ρ_id(z=0) = ρ_id0 = Ω_id0 * ρ_crit0
    rho_id0 = params.Omega_id0

    # z > z_trans: comportamiento tipo materia
    mask_early = z > params.z_trans
    # Normalizar para que sea continuo en z_trans
    rho_at_trans = rho_id0 * (1 + params.z_trans)**3 / (1 + params.epsilon * params.z_trans)
    rho_id[mask_early] = rho_at_trans * ((1 + z[mask_early]) / (1 + params.z_trans))**3

    # z ≤ z_trans: plateau con pendiente ε
    mask_late = ~mask_early
    rho_id[mask_late] = rho_id0 * (1 + params.epsilon * (params.z_trans - z[mask_late]))

    return rho_id


def rho_id_smooth_transition(
    z: np.ndarray,
    params: RhoIdParametricParams
) -> np.ndarray:
    """
    Calcula ρ_id(z) con transición suave (tanh).

    Usa una función tanh para suavizar la transición, evitando
    discontinuidades en derivadas.

    Args:
        z: Array de redshifts
        params: Parámetros del modelo

    Returns:
        Array con ρ_id/ρ_crit0
    """
    z = np.atleast_1d(z)

    # Función de transición suave
    f_trans = 0.5 * (1 + np.tanh((z - params.z_trans) / params.delta_z_trans))

    # Comportamiento temprano (tipo materia)
    rho_early = params.Omega_id0 * (1 + z)**3

    # Comportamiento tardío (plateau)
    rho_late = params.Omega_id0 * (1 + params.epsilon * np.abs(params.z_trans - z))

    # Interpolación suave
    rho_id = f_trans * rho_early + (1 - f_trans) * rho_late

    # Aplicar decay exponencial si gamma > 0
    if params.gamma > 0:
        # El decay actúa desde z_trans hacia z=0
        decay_factor = np.exp(-params.gamma * np.maximum(0, params.z_trans - z))
        rho_id *= decay_factor

    return rho_id


def Omega_id_of_z(
    z: np.ndarray,
    params: RhoIdParametricParams,
    Omega_m0: float = 0.3,
    smooth: bool = True
) -> np.ndarray:
    """
    Calcula la fracción de densidad Ω_id(z) = ρ_id(z) / ρ_crit(z).

    Args:
        z: Array de redshifts
        params: Parámetros de ρ_id
        Omega_m0: Fracción de materia hoy
        smooth: Si usar transición suave

    Returns:
        Array con Ω_id(z)
    """
    if smooth:
        rho_id = rho_id_smooth_transition(z, params)
    else:
        rho_id = rho_id_sharp_transition(z, params, Omega_m0)

    # ρ_crit(z) / ρ_crit0 = E(z)^2 donde E(z) = H(z)/H0
    # Necesitamos E(z) para esto, pero por ahora aproximamos
    # Ω_id(z) ≈ rho_id / (Ω_m0*(1+z)^3 + rho_id)
    rho_m = Omega_m0 * (1 + z)**3

    Omega_id = rho_id / (rho_m + rho_id)

    return Omega_id


def E_squared_with_rho_id(
    z: np.ndarray,
    params: RhoIdParametricParams,
    Omega_m0: float = 0.3,
    Omega_r0: float = 0.0,
    smooth: bool = True
) -> np.ndarray:
    """
    Calcula E(z)^2 = [H(z)/H0]^2 incluyendo ρ_id.

    E(z)^2 = Ω_m0*(1+z)^3 + Ω_r0*(1+z)^4 + ρ_id(z)/ρ_crit0

    Args:
        z: Array de redshifts
        params: Parámetros de ρ_id
        Omega_m0: Fracción de materia
        Omega_r0: Fracción de radiación
        smooth: Si usar transición suave

    Returns:
        Array con E(z)^2
    """
    z = np.atleast_1d(z)

    # Contribución de materia
    E2_m = Omega_m0 * (1 + z)**3

    # Contribución de radiación
    E2_r = Omega_r0 * (1 + z)**4

    # Contribución de ρ_id
    if smooth:
        rho_id = rho_id_smooth_transition(z, params)
    else:
        rho_id = rho_id_sharp_transition(z, params, Omega_m0)

    E2 = E2_m + E2_r + rho_id

    return E2


def H_of_z_with_rho_id(
    z: np.ndarray,
    H0: float,
    params: RhoIdParametricParams,
    Omega_m0: float = 0.3,
    Omega_r0: float = 0.0,
    smooth: bool = True
) -> np.ndarray:
    """
    Calcula H(z) incluyendo la contribución de ρ_id.

    Args:
        z: Array de redshifts
        H0: Constante de Hubble [km/s/Mpc]
        params: Parámetros de ρ_id
        Omega_m0: Fracción de materia
        Omega_r0: Fracción de radiación
        smooth: Si usar transición suave

    Returns:
        Array con H(z) [km/s/Mpc]
    """
    E2 = E_squared_with_rho_id(z, params, Omega_m0, Omega_r0, smooth)
    return H0 * np.sqrt(E2)


def w_eff_id(
    z: np.ndarray,
    params: RhoIdParametricParams,
    dz: float = 0.01
) -> np.ndarray:
    """
    Calcula la ecuación de estado efectiva w_eff = p_id/ρ_id.

    Se deriva numéricamente de la evolución de ρ_id:
        w = -1 - (1/3) * d ln ρ_id / d ln(1+z)

    Args:
        z: Array de redshifts
        params: Parámetros de ρ_id
        dz: Paso para diferenciación numérica

    Returns:
        Array con w_eff(z)
    """
    z = np.atleast_1d(z)
    rho_id = rho_id_smooth_transition(z, params)

    # d ln ρ / d ln(1+z) = (1+z)/ρ * dρ/dz
    # Diferenciación numérica
    rho_plus = rho_id_smooth_transition(z + dz, params)
    rho_minus = rho_id_smooth_transition(z - dz, params)

    d_rho_dz = (rho_plus - rho_minus) / (2 * dz)
    d_ln_rho_d_ln_a = -(1 + z) / rho_id * d_rho_dz

    # w = -1 - (1/3) * d ln ρ / d ln a = -1 + (1/3) * d ln ρ / d ln(1+z)
    w = -1 + d_ln_rho_d_ln_a / 3

    return w


class RhoIdParametricModel:
    """
    Clase que encapsula el modelo paramétrico de ρ_id.

    Proporciona una interfaz unificada para calcular todas las
    cantidades relacionadas con el canal indeterminado.
    """

    def __init__(
        self,
        params: RhoIdParametricParams,
        H0: float = 67.4,
        Omega_m0: float = 0.3,
        Omega_r0: float = 0.0,
        smooth: bool = True
    ):
        """
        Inicializa el modelo.

        Args:
            params: Parámetros de ρ_id
            H0: Constante de Hubble [km/s/Mpc]
            Omega_m0: Fracción de materia
            Omega_r0: Fracción de radiación
            smooth: Si usar transición suave
        """
        self.params = params
        self.H0 = H0
        self.Omega_m0 = Omega_m0
        self.Omega_r0 = Omega_r0
        self.smooth = smooth

    def rho_id(self, z: np.ndarray) -> np.ndarray:
        """Calcula ρ_id(z) / ρ_crit0."""
        if self.smooth:
            return rho_id_smooth_transition(z, self.params)
        else:
            return rho_id_sharp_transition(z, self.params, self.Omega_m0)

    def Omega_id(self, z: np.ndarray) -> np.ndarray:
        """Calcula Ω_id(z)."""
        return Omega_id_of_z(z, self.params, self.Omega_m0, self.smooth)

    def H(self, z: np.ndarray) -> np.ndarray:
        """Calcula H(z) [km/s/Mpc]."""
        return H_of_z_with_rho_id(
            z, self.H0, self.params, self.Omega_m0, self.Omega_r0, self.smooth
        )

    def E(self, z: np.ndarray) -> np.ndarray:
        """Calcula E(z) = H(z)/H0."""
        E2 = E_squared_with_rho_id(
            z, self.params, self.Omega_m0, self.Omega_r0, self.smooth
        )
        return np.sqrt(E2)

    def w_eff(self, z: np.ndarray) -> np.ndarray:
        """Calcula w_eff(z)."""
        return w_eff_id(z, self.params)


# Exportaciones
__all__ = [
    'RhoIdParametricParams',
    'rho_id_sharp_transition',
    'rho_id_smooth_transition',
    'Omega_id_of_z',
    'E_squared_with_rho_id',
    'H_of_z_with_rho_id',
    'w_eff_id',
    'RhoIdParametricModel',
]
