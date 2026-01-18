"""
Observables BAO (Baryon Acoustic Oscillations)
==============================================

Este módulo implementa los observables de oscilaciones acústicas de bariones
para comparación con datos de surveys espectroscópicos (BOSS, eBOSS, DESI, etc.).

Observables principales:
- D_V(z): Distancia de volumen esférico
- D_M(z): Distancia comoving (= d_C para universo plano)
- D_H(z): Distancia de Hubble
- ratios D_M/r_d, D_H/r_d, D_V/r_d

Referencias:
    - Documento de Simulaciones Conjuntas: BAO likelihood
    - BOSS/eBOSS collaboration papers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline

from .distances import C_LIGHT, DistanceCalculator


# Horizonte de sonido fiducial (puede variar según calibración)
R_D_FIDUCIAL = 147.09  # Mpc (Planck 2018)


@dataclass
class BAODataPoint:
    """
    Punto de datos BAO.

    Attributes:
        z_eff: Redshift efectivo
        observable: Tipo de observable ('D_V/r_d', 'D_M/r_d', 'D_H/r_d', etc.)
        value: Valor medido
        error: Error (1σ)
        survey: Nombre del survey (ej. 'BOSS', 'eBOSS', 'DESI')
    """
    z_eff: float
    observable: str
    value: float
    error: float
    survey: str = ''
    correlation: Optional[float] = None  # Correlación con otro observable


@dataclass
class BAODataset:
    """
    Dataset completo de BAO.

    Puede incluir puntos individuales o medidas con matriz de covarianza.
    """
    points: List[BAODataPoint]
    name: str = 'BAO'
    cov_matrix: Optional[np.ndarray] = None  # Matriz de covarianza completa

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def z_values(self) -> np.ndarray:
        return np.array([p.z_eff for p in self.points])

    @property
    def data_values(self) -> np.ndarray:
        return np.array([p.value for p in self.points])

    @property
    def errors(self) -> np.ndarray:
        return np.array([p.error for p in self.points])


def D_H(z: float, H: float) -> float:
    """
    Distancia de Hubble: D_H(z) = c / H(z).

    Args:
        z: Redshift
        H: H(z) [km/s/Mpc]

    Returns:
        D_H [Mpc]
    """
    return C_LIGHT / H


def D_M(z: float, d_C: float) -> float:
    """
    Distancia comoving transversal D_M(z).

    Para universo plano: D_M = d_C

    Args:
        z: Redshift
        d_C: Distancia comoving [Mpc]

    Returns:
        D_M [Mpc]
    """
    # Para universo plano
    return d_C


def D_V(z: float, d_C: float, H: float) -> float:
    """
    Distancia de volumen esférico D_V(z).

    D_V(z) = [z * D_M(z)^2 * D_H(z)]^(1/3)
           = [z * d_C(z)^2 * c / H(z)]^(1/3)

    Esta es la combinación que aparece en análisis isotrópicos de BAO.

    Args:
        z: Redshift
        d_C: Distancia comoving [Mpc]
        H: H(z) [km/s/Mpc]

    Returns:
        D_V [Mpc]
    """
    D_M_val = D_M(z, d_C)
    D_H_val = D_H(z, H)
    return (z * D_M_val**2 * D_H_val) ** (1.0 / 3.0)


def compute_bao_observables(
    z: np.ndarray,
    d_C: np.ndarray,
    H: np.ndarray,
    r_d: float = R_D_FIDUCIAL
) -> Dict[str, np.ndarray]:
    """
    Calcula todos los observables BAO.

    Args:
        z: Array de redshifts
        d_C: Distancia comoving [Mpc]
        H: H(z) [km/s/Mpc]
        r_d: Horizonte de sonido [Mpc]

    Returns:
        Diccionario con D_V, D_M, D_H y sus ratios con r_d
    """
    z = np.atleast_1d(z)
    d_C = np.atleast_1d(d_C)
    H = np.atleast_1d(H)

    D_M_arr = d_C  # Universo plano
    D_H_arr = C_LIGHT / H
    D_V_arr = np.array([D_V(z[i], d_C[i], H[i]) for i in range(len(z))])

    return {
        'z': z,
        'D_M': D_M_arr,
        'D_H': D_H_arr,
        'D_V': D_V_arr,
        'D_M/r_d': D_M_arr / r_d,
        'D_H/r_d': D_H_arr / r_d,
        'D_V/r_d': D_V_arr / r_d,
        'r_d': r_d,
    }


class BAOCalculator:
    """
    Calculador de observables BAO a partir de H(z).
    """

    def __init__(
        self,
        dist_calc: DistanceCalculator,
        H_func=None,
        r_d: float = R_D_FIDUCIAL
    ):
        """
        Inicializa el calculador BAO.

        Args:
            dist_calc: Calculador de distancias
            H_func: Función H(z) (opcional si dist_calc ya tiene tabla)
            r_d: Horizonte de sonido [Mpc]
        """
        self.dist_calc = dist_calc
        self.H_func = H_func
        self.r_d = r_d

    def D_V_over_rd(self, z: float) -> float:
        """Calcula D_V(z)/r_d."""
        d_C = self.dist_calc.d_C(z)
        if self.H_func is not None:
            H_val = self.H_func(z)
        else:
            # Interpolar de la tabla
            H_interp = CubicSpline(
                self.dist_calc.z_grid,
                self.dist_calc.H_grid
            )
            H_val = H_interp(z)
        D_V_val = D_V(z, d_C, H_val)
        return D_V_val / self.r_d

    def D_M_over_rd(self, z: float) -> float:
        """Calcula D_M(z)/r_d."""
        d_C = self.dist_calc.d_C(z)
        return d_C / self.r_d

    def D_H_over_rd(self, z: float) -> float:
        """Calcula D_H(z)/r_d."""
        if self.H_func is not None:
            H_val = self.H_func(z)
        else:
            H_interp = CubicSpline(
                self.dist_calc.z_grid,
                self.dist_calc.H_grid
            )
            H_val = H_interp(z)
        D_H_val = D_H(z, H_val)
        return D_H_val / self.r_d

    def compute_for_dataset(self, dataset: BAODataset) -> Dict[str, np.ndarray]:
        """
        Calcula predicciones del modelo para un dataset BAO.

        Args:
            dataset: Dataset BAO con puntos de datos

        Returns:
            Diccionario con predicciones en cada z_eff
        """
        predictions = {}

        for point in dataset.points:
            z = point.z_eff

            if point.observable == 'D_V/r_d':
                predictions[f'{point.observable}_{z}'] = self.D_V_over_rd(z)
            elif point.observable == 'D_M/r_d':
                predictions[f'{point.observable}_{z}'] = self.D_M_over_rd(z)
            elif point.observable == 'D_H/r_d':
                predictions[f'{point.observable}_{z}'] = self.D_H_over_rd(z)

        return predictions


# Datasets predefinidos
def get_boss_dr12_data() -> BAODataset:
    """
    Retorna dataset BOSS DR12 BAO.

    Valores de Alam et al. (2017).
    """
    points = [
        BAODataPoint(0.38, 'D_M/r_d', 10.27, 0.15, 'BOSS DR12'),
        BAODataPoint(0.38, 'D_H/r_d', 25.00, 0.76, 'BOSS DR12'),
        BAODataPoint(0.51, 'D_M/r_d', 13.38, 0.18, 'BOSS DR12'),
        BAODataPoint(0.51, 'D_H/r_d', 22.33, 0.58, 'BOSS DR12'),
        BAODataPoint(0.61, 'D_M/r_d', 15.45, 0.21, 'BOSS DR12'),
        BAODataPoint(0.61, 'D_H/r_d', 20.85, 0.53, 'BOSS DR12'),
    ]
    return BAODataset(points=points, name='BOSS DR12')


def get_eboss_dr16_data() -> BAODataset:
    """
    Retorna dataset eBOSS DR16 BAO.

    Valores aproximados de eBOSS collaboration.
    """
    points = [
        # LRG
        BAODataPoint(0.70, 'D_M/r_d', 17.86, 0.33, 'eBOSS LRG'),
        BAODataPoint(0.70, 'D_H/r_d', 19.33, 0.53, 'eBOSS LRG'),
        # ELG
        BAODataPoint(0.85, 'D_M/r_d', 19.50, 1.0, 'eBOSS ELG'),
        # QSO
        BAODataPoint(1.48, 'D_M/r_d', 30.21, 0.79, 'eBOSS QSO'),
        BAODataPoint(1.48, 'D_H/r_d', 13.23, 0.47, 'eBOSS QSO'),
        # Lyman-alpha
        BAODataPoint(2.33, 'D_M/r_d', 37.6, 1.9, 'eBOSS Lya'),
        BAODataPoint(2.33, 'D_H/r_d', 8.93, 0.28, 'eBOSS Lya'),
    ]
    return BAODataset(points=points, name='eBOSS DR16')


def get_6dFGS_data() -> BAODataset:
    """Retorna dataset 6dFGS BAO (bajo z)."""
    points = [
        BAODataPoint(0.106, 'D_V/r_d', 2.98, 0.13, '6dFGS'),
    ]
    return BAODataset(points=points, name='6dFGS')


def get_combined_bao_data() -> BAODataset:
    """Combina todos los datasets BAO disponibles."""
    boss = get_boss_dr12_data()
    eboss = get_eboss_dr16_data()
    sixdf = get_6dFGS_data()

    all_points = boss.points + eboss.points + sixdf.points
    return BAODataset(points=all_points, name='Combined BAO')


# Exportaciones
__all__ = [
    'R_D_FIDUCIAL',
    'BAODataPoint',
    'BAODataset',
    'D_H',
    'D_M',
    'D_V',
    'compute_bao_observables',
    'BAOCalculator',
    'get_boss_dr12_data',
    'get_eboss_dr16_data',
    'get_6dFGS_data',
    'get_combined_bao_data',
]
