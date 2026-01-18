"""
Ecuación de Estado del Sector Oscuro
====================================

Este módulo implementa las ecuaciones de estado (EoS) para los canales
oscuros del MCMC: w_DE(z), c_s^2, y parametrizaciones derivadas.

La EoS total del sector oscuro se define como:
    w_DE = p_DE / ρ_DE = (p_id + p_lat) / (ρ_id + ρ_lat)

Para perturbaciones, se requiere también la velocidad del sonido:
    c_s^2 = δp / δρ

En el MCMC se recomienda c_s^2 ≈ 1 para estabilidad.

Referencias:
    - Tratado MCMC Maestro: ecuación de estado
    - Documentación CLASS/CAMB: w(z) y c_s^2 para fluidos
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class EoSParams:
    """
    Parámetros para la ecuación de estado w_DE(z).

    Soporta varias parametrizaciones:
    - Constante: w(z) = w0
    - CPL: w(z) = w0 + wa * z/(1+z)
    - MCMC: reconstruida desde evolución de ρ_id y ρ_lat

    Attributes:
        w0: Valor de w hoy (z=0)
        wa: Parámetro de evolución (CPL)
        cs2: Velocidad del sonido al cuadrado
        parametrization: Tipo de parametrización
    """
    w0: float = -1.0            # EoS hoy
    wa: float = 0.0             # Evolución (CPL)
    cs2: float = 1.0            # Velocidad del sonido al cuadrado
    parametrization: str = 'constant'  # 'constant', 'cpl', 'mcmc'

    def to_dict(self) -> Dict:
        """Convierte a diccionario."""
        return vars(self).copy()


def w_constant(z: np.ndarray, w0: float) -> np.ndarray:
    """
    Ecuación de estado constante: w(z) = w0.

    Args:
        z: Array de redshifts
        w0: Valor constante de w

    Returns:
        Array con w(z) = w0
    """
    return np.full_like(z, w0, dtype=float)


def w_cpl(z: np.ndarray, w0: float, wa: float) -> np.ndarray:
    """
    Parametrización CPL (Chevallier-Polarski-Linder).

    w(z) = w0 + wa * z / (1 + z)

    En el límite:
    - z → 0: w → w0
    - z → ∞: w → w0 + wa

    Args:
        z: Array de redshifts
        w0: Valor de w en z=0
        wa: Parámetro de evolución

    Returns:
        Array con w(z)
    """
    z = np.atleast_1d(z)
    return w0 + wa * z / (1.0 + z)


def w_mcmc_transition(
    z: np.ndarray,
    w_early: float = 0.0,
    w_late: float = -1.0,
    z_trans: float = 0.5,
    delta_z: float = 0.1
) -> np.ndarray:
    """
    EoS con transición suave para el MCMC.

    w(z) transita de w_early (z >> z_trans) a w_late (z << z_trans).
    Esto representa la transición de comportamiento tipo materia
    a comportamiento tipo constante cosmológica.

    Args:
        z: Array de redshifts
        w_early: EoS en z alto (típicamente 0 para materia)
        w_late: EoS en z bajo (típicamente -1 para Λ)
        z_trans: Redshift de transición
        delta_z: Ancho de la transición

    Returns:
        Array con w(z)
    """
    z = np.atleast_1d(z)

    # Función de transición suave (tanh)
    f = 0.5 * (1 + np.tanh((z - z_trans) / delta_z))

    # Interpolación: w_early en z alto, w_late en z bajo
    w = f * w_early + (1 - f) * w_late

    return w


def w_from_rho_evolution(
    z: np.ndarray,
    rho: np.ndarray,
    dz: float = 0.01
) -> np.ndarray:
    """
    Reconstruye w(z) desde la evolución de ρ(z).

    Usando la ecuación de conservación:
    dρ/dz = 3(1+w)ρ / (1+z)

    =>  w = -1 + (1+z)/(3ρ) * dρ/dz

    Args:
        z: Array de redshifts
        rho: Densidad de energía
        dz: Paso para diferenciación numérica (si se usa)

    Returns:
        Array con w(z)
    """
    z = np.atleast_1d(z)
    rho = np.atleast_1d(rho)

    # Derivada numérica dρ/dz
    d_rho_dz = np.gradient(rho, z)

    # w = -1 + (1+z)/(3ρ) * dρ/dz
    # Evitar división por cero
    mask = np.abs(rho) > 1e-30
    w = np.full_like(z, -1.0, dtype=float)
    w[mask] = -1.0 + (1.0 + z[mask]) / (3.0 * rho[mask]) * d_rho_dz[mask]

    return w


def pressure_from_rho_and_w(
    rho: np.ndarray,
    w: np.ndarray
) -> np.ndarray:
    """
    Calcula la presión: p = w * ρ.

    Args:
        rho: Densidad de energía
        w: Ecuación de estado

    Returns:
        Array con p
    """
    return w * rho


def cs2_from_w(
    w: np.ndarray,
    adiabatic: bool = True
) -> np.ndarray:
    """
    Calcula c_s^2 desde w.

    Para un fluido barotropico perfecto adiabático: c_s^2 = w.
    Para el MCMC con c_s^2 fija: c_s^2 = constante ≈ 1.

    Args:
        w: Ecuación de estado
        adiabatic: Si True, usa c_s^2 = w; si False, usa c_s^2 = 1

    Returns:
        Array con c_s^2
    """
    if adiabatic:
        return w.copy()
    else:
        return np.ones_like(w)


def rho_evolution_from_w(
    z: np.ndarray,
    w: np.ndarray,
    rho_0: float
) -> np.ndarray:
    """
    Calcula ρ(z) desde w(z) integrando la ecuación de conservación.

    dρ/dz = 3(1+w)ρ / (1+z)

    Args:
        z: Array de redshifts (ordenado de z=0 hacia arriba)
        w: Ecuación de estado w(z)
        rho_0: Densidad hoy (z=0)

    Returns:
        Array con ρ(z) / ρ_crit0
    """
    z = np.atleast_1d(z)
    w = np.atleast_1d(w)

    # Para w constante: ρ ∝ (1+z)^{3(1+w)}
    # Para w variable: integrar numéricamente

    # Integral: ln(ρ/ρ_0) = 3 ∫_0^z (1+w(z'))/(1+z') dz'
    integrand = 3.0 * (1.0 + w) / (1.0 + z)

    # Ordenar por z creciente para la integral
    idx_sort = np.argsort(z)
    z_sorted = z[idx_sort]
    integrand_sorted = integrand[idx_sort]

    # Integral cumulativa
    from scipy.integrate import cumulative_trapezoid
    ln_rho_ratio = np.zeros_like(z_sorted)
    ln_rho_ratio[1:] = cumulative_trapezoid(integrand_sorted, z_sorted)

    # ρ(z) = ρ_0 * exp(integral)
    rho_sorted = rho_0 * np.exp(ln_rho_ratio)

    # Reordenar al orden original
    rho = np.empty_like(z)
    rho[idx_sort] = rho_sorted

    return rho


class DarkEnergyEoS:
    """
    Clase que encapsula la ecuación de estado del sector oscuro.

    Proporciona una interfaz unificada para evaluar w(z), c_s^2(z),
    y otras cantidades relacionadas.
    """

    def __init__(
        self,
        params: EoSParams = None,
        z_table: np.ndarray = None,
        w_table: np.ndarray = None
    ):
        """
        Inicializa la EoS.

        Puede inicializarse con parámetros (para parametrizaciones estándar)
        o con tablas (para EoS reconstruida desde datos).

        Args:
            params: Parámetros de la EoS
            z_table: Array de z para EoS tabular
            w_table: Array de w para EoS tabular
        """
        self.params = params if params is not None else EoSParams()
        self.z_table = z_table
        self.w_table = w_table

        if z_table is not None and w_table is not None:
            self._w_interp = CubicSpline(z_table, w_table)
            self._tabular = True
        else:
            self._tabular = False

    def w(self, z: np.ndarray) -> np.ndarray:
        """
        Evalúa w(z).

        Args:
            z: Redshift(s)

        Returns:
            Array con w(z)
        """
        z = np.atleast_1d(z)

        if self._tabular:
            return self._w_interp(z)

        if self.params.parametrization == 'constant':
            return w_constant(z, self.params.w0)
        elif self.params.parametrization == 'cpl':
            return w_cpl(z, self.params.w0, self.params.wa)
        else:
            raise ValueError(f"Parametrización desconocida: {self.params.parametrization}")

    def cs2(self, z: np.ndarray) -> np.ndarray:
        """
        Evalúa c_s^2(z).

        En el MCMC se usa típicamente c_s^2 = 1 para estabilidad.

        Args:
            z: Redshift(s)

        Returns:
            Array con c_s^2(z)
        """
        z = np.atleast_1d(z)
        return np.full_like(z, self.params.cs2, dtype=float)

    def rho(self, z: np.ndarray, rho_0: float = 1.0) -> np.ndarray:
        """
        Calcula ρ(z) consistente con w(z).

        Args:
            z: Redshift(s)
            rho_0: Densidad hoy

        Returns:
            Array con ρ(z)
        """
        z = np.atleast_1d(z)
        w = self.w(z)
        return rho_evolution_from_w(z, w, rho_0)

    @classmethod
    def from_rho_table(
        cls,
        z: np.ndarray,
        rho: np.ndarray,
        cs2: float = 1.0
    ) -> 'DarkEnergyEoS':
        """
        Construye EoS desde tabla de ρ(z).

        Args:
            z: Array de redshifts
            rho: Array de densidades
            cs2: Velocidad del sonido al cuadrado

        Returns:
            DarkEnergyEoS con w(z) reconstruida
        """
        w = w_from_rho_evolution(z, rho)
        params = EoSParams(cs2=cs2, parametrization='mcmc')
        return cls(params=params, z_table=z, w_table=w)


# Funciones de conveniencia
def get_lambda_cdm_eos() -> DarkEnergyEoS:
    """Retorna EoS de ΛCDM: w = -1, c_s^2 = 1."""
    return DarkEnergyEoS(EoSParams(w0=-1.0, wa=0.0, cs2=1.0))


def get_quintessence_eos(w0: float = -0.9) -> DarkEnergyEoS:
    """Retorna EoS tipo quintaesencia: w > -1."""
    return DarkEnergyEoS(EoSParams(w0=w0, wa=0.0, cs2=1.0))


def get_phantom_eos(w0: float = -1.1) -> DarkEnergyEoS:
    """Retorna EoS tipo phantom: w < -1."""
    return DarkEnergyEoS(EoSParams(w0=w0, wa=0.0, cs2=1.0))


# Exportaciones
__all__ = [
    'EoSParams',
    'w_constant',
    'w_cpl',
    'w_mcmc_transition',
    'w_from_rho_evolution',
    'pressure_from_rho_and_w',
    'cs2_from_w',
    'rho_evolution_from_w',
    'DarkEnergyEoS',
    'get_lambda_cdm_eos',
    'get_quintessence_eos',
    'get_phantom_eos',
]
