"""
Distancias Cosmológicas
=======================

Este módulo implementa el cálculo de distancias cosmológicas para el MCMC:
- Distancia comoving d_C(z)
- Distancia de diámetro angular d_A(z)
- Distancia de luminosidad d_L(z)
- Módulo de distancia μ(z)

Las distancias se calculan integrando:
    d_C(z) = c ∫_0^z dz' / H(z')

Referencias:
    - Documento de Simulaciones Conjuntas
    - Hogg (1999), "Distance measures in cosmology"
"""

from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import CubicSpline


# Constantes físicas
C_LIGHT = 299792.458  # km/s (velocidad de la luz)


def comoving_distance(
    z: np.ndarray,
    H_func: Callable[[float], float],
    z_grid: Optional[np.ndarray] = None,
    H_grid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calcula la distancia comoving d_C(z).

    d_C(z) = c ∫_0^z dz' / H(z')   [Mpc]

    Args:
        z: Array de redshifts donde evaluar
        H_func: Función H(z) [km/s/Mpc]
        z_grid: Grid de z para método de integración con tabla (opcional)
        H_grid: Valores de H en z_grid (opcional)

    Returns:
        Array con d_C(z) [Mpc]
    """
    z = np.atleast_1d(z)

    if z_grid is not None and H_grid is not None:
        # Método con interpolación
        return _comoving_distance_table(z, z_grid, H_grid)
    else:
        # Método con integración directa
        return _comoving_distance_quad(z, H_func)


def _comoving_distance_quad(
    z: np.ndarray,
    H_func: Callable[[float], float]
) -> np.ndarray:
    """Calcula d_C usando integración con quad."""
    d_C = np.zeros_like(z, dtype=float)

    def integrand(zp):
        return C_LIGHT / H_func(zp)

    for i, zi in enumerate(z):
        if zi <= 0:
            d_C[i] = 0.0
        else:
            result, _ = quad(integrand, 0, zi)
            d_C[i] = result

    return d_C


def _comoving_distance_table(
    z: np.ndarray,
    z_grid: np.ndarray,
    H_grid: np.ndarray
) -> np.ndarray:
    """Calcula d_C usando integración con tabla precalculada."""
    # Asegurar que z_grid está ordenado
    idx_sort = np.argsort(z_grid)
    z_sorted = z_grid[idx_sort]
    H_sorted = H_grid[idx_sort]

    # Integrando: c/H(z)
    integrand = C_LIGHT / H_sorted

    # Integral cumulativa
    d_C_grid = np.zeros_like(z_sorted)
    d_C_grid[1:] = cumulative_trapezoid(integrand, z_sorted)

    # Interpolar a los z solicitados
    d_C_interp = CubicSpline(z_sorted, d_C_grid)

    return d_C_interp(z)


def angular_diameter_distance(
    z: np.ndarray,
    d_C: np.ndarray
) -> np.ndarray:
    """
    Calcula la distancia de diámetro angular d_A(z).

    d_A(z) = d_C(z) / (1 + z)   [Mpc]

    Args:
        z: Array de redshifts
        d_C: Distancia comoving [Mpc]

    Returns:
        Array con d_A(z) [Mpc]
    """
    return d_C / (1.0 + z)


def luminosity_distance(
    z: np.ndarray,
    d_C: np.ndarray
) -> np.ndarray:
    """
    Calcula la distancia de luminosidad d_L(z).

    d_L(z) = (1 + z) * d_C(z)   [Mpc]

    Args:
        z: Array de redshifts
        d_C: Distancia comoving [Mpc]

    Returns:
        Array con d_L(z) [Mpc]
    """
    return (1.0 + z) * d_C


def distance_modulus(d_L: np.ndarray) -> np.ndarray:
    """
    Calcula el módulo de distancia μ(z).

    μ = 5 * log10(d_L / 10 pc) = 5 * log10(d_L) + 25

    donde d_L está en Mpc.

    Args:
        d_L: Distancia de luminosidad [Mpc]

    Returns:
        Array con μ
    """
    # d_L en Mpc, 10 pc = 1e-5 Mpc
    # μ = 5 * log10(d_L / 1e-5) = 5 * log10(d_L) + 25
    return 5.0 * np.log10(d_L) + 25.0


def hubble_distance(H0: float) -> float:
    """
    Distancia de Hubble: d_H = c / H0.

    Args:
        H0: Constante de Hubble [km/s/Mpc]

    Returns:
        d_H [Mpc]
    """
    return C_LIGHT / H0


class DistanceCalculator:
    """
    Calculador de distancias cosmológicas.

    Proporciona una interfaz unificada para calcular todas las
    distancias a partir de H(z).
    """

    def __init__(
        self,
        H_func: Callable[[float], float] = None,
        z_grid: np.ndarray = None,
        H_grid: np.ndarray = None,
        H0: float = 67.4
    ):
        """
        Inicializa el calculador.

        Args:
            H_func: Función H(z) [km/s/Mpc]
            z_grid: Grid de z para método con tabla
            H_grid: Valores de H en z_grid
            H0: Constante de Hubble (para normalización)
        """
        self.H0 = H0
        self.H_func = H_func
        self.z_grid = z_grid
        self.H_grid = H_grid

        if H_func is None and (z_grid is None or H_grid is None):
            raise ValueError("Debe proporcionar H_func o (z_grid, H_grid)")

        # Precalcular si hay tabla
        if z_grid is not None and H_grid is not None:
            self._precompute_distances()

    def _precompute_distances(self):
        """Precalcula distancias en la grid."""
        self._d_C_grid = comoving_distance(
            self.z_grid, None, self.z_grid, self.H_grid
        )
        self._d_A_grid = angular_diameter_distance(self.z_grid, self._d_C_grid)
        self._d_L_grid = luminosity_distance(self.z_grid, self._d_C_grid)
        self._mu_grid = distance_modulus(self._d_L_grid)

        # Interpoladores
        self._d_C_interp = CubicSpline(self.z_grid, self._d_C_grid)
        self._d_A_interp = CubicSpline(self.z_grid, self._d_A_grid)
        self._d_L_interp = CubicSpline(self.z_grid, self._d_L_grid)
        self._mu_interp = CubicSpline(self.z_grid, self._mu_grid)

    def d_C(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Distancia comoving [Mpc]."""
        z = np.atleast_1d(z)
        if hasattr(self, '_d_C_interp'):
            result = self._d_C_interp(z)
        else:
            result = comoving_distance(z, self.H_func)
        return float(result) if result.size == 1 else result

    def d_A(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Distancia de diámetro angular [Mpc]."""
        z = np.atleast_1d(z)
        if hasattr(self, '_d_A_interp'):
            result = self._d_A_interp(z)
        else:
            d_C = comoving_distance(z, self.H_func)
            result = angular_diameter_distance(z, d_C)
        return float(result) if result.size == 1 else result

    def d_L(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Distancia de luminosidad [Mpc]."""
        z = np.atleast_1d(z)
        if hasattr(self, '_d_L_interp'):
            result = self._d_L_interp(z)
        else:
            d_C = comoving_distance(z, self.H_func)
            result = luminosity_distance(z, d_C)
        return float(result) if result.size == 1 else result

    def mu(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Módulo de distancia."""
        z = np.atleast_1d(z)
        if hasattr(self, '_mu_interp'):
            result = self._mu_interp(z)
        else:
            d_L = self.d_L(z)
            result = distance_modulus(d_L)
        return float(result) if result.size == 1 else result

    def d_H(self) -> float:
        """Distancia de Hubble [Mpc]."""
        return hubble_distance(self.H0)

    def all_distances(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcula todas las distancias.

        Args:
            z: Array de redshifts

        Returns:
            Diccionario con d_C, d_A, d_L, mu
        """
        d_C = self.d_C(z)
        return {
            'z': z,
            'd_C': d_C,
            'd_A': angular_diameter_distance(z, d_C),
            'd_L': luminosity_distance(z, d_C),
            'mu': distance_modulus(luminosity_distance(z, d_C)),
        }


# Funciones de conveniencia para ΛCDM
def H_lcdm(z: float, H0: float = 67.4, Omega_m: float = 0.3) -> float:
    """H(z) para ΛCDM plano."""
    Omega_L = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)


def get_lcdm_distances(
    z: np.ndarray,
    H0: float = 67.4,
    Omega_m: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Calcula distancias para ΛCDM.

    Args:
        z: Array de redshifts
        H0: Constante de Hubble
        Omega_m: Fracción de materia

    Returns:
        Diccionario con todas las distancias
    """
    def H_func(z_val):
        return H_lcdm(z_val, H0, Omega_m)

    calc = DistanceCalculator(H_func=H_func, H0=H0)
    return calc.all_distances(z)


# Exportaciones
__all__ = [
    'C_LIGHT',
    'comoving_distance',
    'angular_diameter_distance',
    'luminosity_distance',
    'distance_modulus',
    'hubble_distance',
    'DistanceCalculator',
    'H_lcdm',
    'get_lcdm_distances',
]
