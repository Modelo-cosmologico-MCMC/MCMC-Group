"""
Canal Latente ρ_lat
===================

Este módulo implementa el canal de energía latente ρ_lat del MCMC.

El canal latente representa la energía "sellada" que aún no ha sido
liberada. Su dinámica está gobernada por:

    dρ_lat/dS = κ_lat(S) - η_lat(S)

donde:
- κ_lat(S): tasa de acumulación (energía que se "sella")
- η_lat(S): tasa de liberación (energía que se convierte)

Restricción importante:
- La fracción latente debe ser pequeña en recombinación
  para no afectar el CMB significativamente.

Referencias:
    - Tratado MCMC Maestro: canal latente
    - Restricciones de early dark energy del CMB
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline


@dataclass
class RhoLatParams:
    """
    Parámetros del canal latente ρ_lat.

    Attributes:
        rho_lat_init: Valor inicial en S_min
        kappa_amplitude: Amplitud de la tasa de acumulación
        kappa_S_center: Centro de la región de acumulación
        kappa_width: Ancho de la región de acumulación
        eta_amplitude: Amplitud de la tasa de liberación
        eta_S_onset: Inicio de la liberación (en S)
        eta_rate: Tasa de crecimiento de la liberación
        max_fraction_recomb: Fracción máxima permitida en recombinación
    """
    rho_lat_init: float = 0.001     # Condición inicial (muy pequeña)
    kappa_amplitude: float = 0.5    # Amplitud de acumulación
    kappa_S_center: float = 0.3     # Centro de acumulación
    kappa_width: float = 0.1        # Ancho de acumulación
    eta_amplitude: float = 0.3      # Amplitud de liberación
    eta_S_onset: float = 0.7        # Inicio de liberación
    eta_rate: float = 5.0           # Tasa de crecimiento de liberación
    max_fraction_recomb: float = 0.05  # Restricción CMB

    def to_dict(self) -> Dict[str, float]:
        """Convierte a diccionario."""
        return vars(self).copy()

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'RhoLatParams':
        """Construye desde diccionario."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def kappa_lat(S: np.ndarray, params: RhoLatParams) -> np.ndarray:
    """
    Tasa de acumulación κ_lat(S).

    Representa la tasa a la que la energía se "sella" en el canal latente.
    Modelada como una gaussiana centrada en kappa_S_center.

    Args:
        S: Array de variable entrópica
        params: Parámetros del modelo

    Returns:
        Array con κ_lat(S)
    """
    return params.kappa_amplitude * np.exp(
        -0.5 * ((S - params.kappa_S_center) / params.kappa_width)**2
    )


def eta_lat(S: np.ndarray, params: RhoLatParams) -> np.ndarray:
    """
    Tasa de liberación η_lat(S).

    Representa la tasa a la que la energía latente se libera/convierte.
    Modelada con una función sigmoide que "enciende" después de eta_S_onset.

    Args:
        S: Array de variable entrópica
        params: Parámetros del modelo

    Returns:
        Array con η_lat(S)
    """
    # Sigmoide: crece después de eta_S_onset
    sigmoid = 1.0 / (1.0 + np.exp(-params.eta_rate * (S - params.eta_S_onset)))
    return params.eta_amplitude * sigmoid


def solve_rho_lat(
    S: np.ndarray,
    params: RhoLatParams,
    return_rates: bool = False
) -> np.ndarray:
    """
    Resuelve la ecuación de evolución para ρ_lat.

    dρ_lat/dS = κ_lat(S) - η_lat(S)

    Esta es una integración directa.

    Args:
        S: Array de variable entrópica
        params: Parámetros del modelo
        return_rates: Si True, retorna también κ y η

    Returns:
        Array con ρ_lat(S), o tupla (ρ_lat, κ, η) si return_rates=True
    """
    # Calcular tasas
    kappa = kappa_lat(S, params)
    eta = eta_lat(S, params)

    # Término fuente neto
    net_source = kappa - eta

    # Integrar desde S_min
    rho_lat = np.zeros_like(S)
    rho_lat[0] = params.rho_lat_init

    # Integración cumulativa
    integral = cumulative_trapezoid(net_source, S, initial=0)
    rho_lat = params.rho_lat_init + integral

    # ρ_lat no puede ser negativo
    rho_lat = np.maximum(rho_lat, 0.0)

    if return_rates:
        return rho_lat, kappa, eta
    return rho_lat


def check_recomb_constraint(
    rho_lat: np.ndarray,
    S: np.ndarray,
    z: np.ndarray,
    rho_total: np.ndarray,
    params: RhoLatParams,
    z_recomb: float = 1089.0
) -> Tuple[bool, float]:
    """
    Verifica la restricción de fracción latente en recombinación.

    La fracción Ω_lat(z_recomb) = ρ_lat/ρ_total debe ser pequeña
    para no afectar el CMB.

    Args:
        rho_lat: Densidad de ρ_lat
        S: Rejilla entrópica
        z: Redshift
        rho_total: Densidad total
        params: Parámetros (contiene max_fraction_recomb)
        z_recomb: Redshift de recombinación

    Returns:
        Tupla (passed, fraction_at_recomb)
    """
    # Encontrar el punto más cercano a recombinación
    idx_recomb = np.argmin(np.abs(z - z_recomb))

    # Fracción en recombinación
    fraction = rho_lat[idx_recomb] / rho_total[idx_recomb]

    passed = fraction < params.max_fraction_recomb

    return passed, fraction


def rho_de_total(
    rho_id: np.ndarray,
    rho_lat: np.ndarray
) -> np.ndarray:
    """
    Densidad total de energía oscura: ρ_DE = ρ_id + ρ_lat.

    Args:
        rho_id: Canal indeterminado
        rho_lat: Canal latente

    Returns:
        Array con ρ_DE(S)
    """
    return rho_id + rho_lat


def Omega_lat_of_z(
    rho_lat: np.ndarray,
    z: np.ndarray,
    Omega_m0: float = 0.3,
    rho_id: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula la fracción Ω_lat(z).

    Args:
        rho_lat: Canal latente
        z: Redshift
        Omega_m0: Fracción de materia hoy
        rho_id: Canal indeterminado (opcional)

    Returns:
        Array con Ω_lat(z)
    """
    rho_m = Omega_m0 * (1 + z)**3

    if rho_id is not None:
        rho_total = rho_m + rho_id + rho_lat
    else:
        rho_total = rho_m + rho_lat

    return rho_lat / rho_total


class RhoLatModel:
    """
    Clase que encapsula el modelo del canal latente.
    """

    def __init__(
        self,
        S: np.ndarray,
        params: RhoLatParams,
        z: Optional[np.ndarray] = None,
        rho_id: Optional[np.ndarray] = None
    ):
        """
        Inicializa y resuelve el modelo.

        Args:
            S: Rejilla entrópica
            params: Parámetros del modelo
            z: Redshift (opcional)
            rho_id: Canal indeterminado (opcional)
        """
        self.S = S
        self.params = params
        self.z = z
        self.rho_id = rho_id

        # Resolver ecuación de evolución
        self.rho_lat, self.kappa, self.eta = solve_rho_lat(
            S, params, return_rates=True
        )

        # Construir interpolador
        self._rho_lat_of_S = CubicSpline(S, self.rho_lat)

        if z is not None:
            idx = np.argsort(z)
            self._rho_lat_of_z = CubicSpline(z[idx], self.rho_lat[idx])

    def rho_lat_at_S(self, S: float) -> float:
        """Evalúa ρ_lat en S dado."""
        return float(self._rho_lat_of_S(S))

    def rho_lat_at_z(self, z: float) -> float:
        """Evalúa ρ_lat en z dado."""
        if self.z is None:
            raise ValueError("z no fue proporcionado")
        return float(self._rho_lat_of_z(z))

    @property
    def rho_de(self) -> np.ndarray:
        """ρ_DE = ρ_id + ρ_lat si ρ_id está disponible."""
        if self.rho_id is not None:
            return self.rho_id + self.rho_lat
        return self.rho_lat


# Exportaciones
__all__ = [
    'RhoLatParams',
    'kappa_lat',
    'eta_lat',
    'solve_rho_lat',
    'check_recomb_constraint',
    'rho_de_total',
    'Omega_lat_of_z',
    'RhoLatModel',
]
