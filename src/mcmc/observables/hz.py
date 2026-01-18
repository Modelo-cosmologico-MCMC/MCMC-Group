"""
Observables H(z) - Cosmic Chronometers
======================================

Este módulo implementa el manejo de datos de H(z) medidos directamente,
principalmente de cosmic chronometers (relojes cósmicos).

Los cosmic chronometers miden H(z) usando la relación de edades de galaxias:
    H(z) = -1/(1+z) * dz/dt

Esto proporciona medidas directas de H(z) independientes de la distancia.

Referencias:
    - Documento de Simulaciones Conjuntas: H(z) likelihood
    - Jimenez & Loeb (2002), Moresco et al. (2012, 2016)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class HzDataPoint:
    """
    Punto de datos H(z).

    Attributes:
        z: Redshift
        H: H(z) medido [km/s/Mpc]
        sigma_H: Error (1σ) [km/s/Mpc]
        method: Método de medición ('CC' para cosmic chronometers, 'BAO', etc.)
        reference: Referencia bibliográfica
    """
    z: float
    H: float
    sigma_H: float
    method: str = 'CC'
    reference: str = ''


@dataclass
class HzDataset:
    """
    Dataset de medidas H(z).
    """
    points: List[HzDataPoint]
    name: str = 'H(z)'

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def z_values(self) -> np.ndarray:
        return np.array([p.z for p in self.points])

    @property
    def H_values(self) -> np.ndarray:
        return np.array([p.H for p in self.points])

    @property
    def errors(self) -> np.ndarray:
        return np.array([p.sigma_H for p in self.points])


def chi2_Hz(
    H_model: np.ndarray,
    H_data: np.ndarray,
    sigma_H: np.ndarray
) -> float:
    """
    Calcula χ² para datos de H(z).

    χ² = Σ [(H_model - H_data) / σ_H]²

    Args:
        H_model: Predicciones del modelo
        H_data: Datos observados
        sigma_H: Errores

    Returns:
        χ²
    """
    residuals = (H_model - H_data) / sigma_H
    return np.sum(residuals**2)


def evaluate_Hz_model(
    z_data: np.ndarray,
    H_func,
    dataset: HzDataset
) -> Dict[str, float]:
    """
    Evalúa un modelo contra datos de H(z).

    Args:
        z_data: Redshifts de los datos
        H_func: Función H(z) del modelo
        dataset: Dataset de H(z)

    Returns:
        Diccionario con χ² y otros estadísticos
    """
    H_model = np.array([H_func(z) for z in z_data])
    H_data = dataset.H_values
    sigma_H = dataset.errors

    chi2 = chi2_Hz(H_model, H_data, sigma_H)

    return {
        'chi2': chi2,
        'n_points': len(z_data),
        'chi2_reduced': chi2 / len(z_data),
    }


# Compilación de datos de cosmic chronometers
def get_cosmic_chronometers_data() -> HzDataset:
    """
    Retorna compilación de datos de cosmic chronometers.

    Datos de Moresco et al. (2012, 2016) y otras fuentes.
    """
    points = [
        # Moresco et al. 2012
        HzDataPoint(0.070, 69.0, 19.6, 'CC', 'Moresco+12'),
        HzDataPoint(0.090, 69.0, 12.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.120, 68.6, 26.2, 'CC', 'Moresco+12'),
        HzDataPoint(0.170, 83.0, 8.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.179, 75.0, 4.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.199, 75.0, 5.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.200, 72.9, 29.6, 'CC', 'Moresco+12'),
        HzDataPoint(0.270, 77.0, 14.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.280, 88.8, 36.6, 'CC', 'Moresco+12'),
        HzDataPoint(0.352, 83.0, 14.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.3802, 83.0, 13.5, 'CC', 'Moresco+16'),
        HzDataPoint(0.400, 95.0, 17.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.4004, 77.0, 10.2, 'CC', 'Moresco+16'),
        HzDataPoint(0.4247, 87.1, 11.2, 'CC', 'Moresco+16'),
        HzDataPoint(0.4497, 92.8, 12.9, 'CC', 'Moresco+16'),
        HzDataPoint(0.4783, 80.9, 9.0, 'CC', 'Moresco+16'),
        HzDataPoint(0.480, 97.0, 62.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.593, 104.0, 13.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.680, 92.0, 8.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.781, 105.0, 12.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.875, 125.0, 17.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.880, 90.0, 40.0, 'CC', 'Moresco+12'),
        HzDataPoint(0.900, 117.0, 23.0, 'CC', 'Moresco+12'),
        HzDataPoint(1.037, 154.0, 20.0, 'CC', 'Moresco+12'),
        # Moresco 2015 - alto z
        HzDataPoint(1.300, 168.0, 17.0, 'CC', 'Moresco+15'),
        HzDataPoint(1.363, 160.0, 33.6, 'CC', 'Moresco+15'),
        HzDataPoint(1.430, 177.0, 18.0, 'CC', 'Moresco+15'),
        HzDataPoint(1.530, 140.0, 14.0, 'CC', 'Moresco+15'),
        HzDataPoint(1.750, 202.0, 40.0, 'CC', 'Moresco+15'),
        HzDataPoint(1.965, 186.5, 50.4, 'CC', 'Moresco+15'),
    ]
    return HzDataset(points=points, name='Cosmic Chronometers')


def get_Hz_from_bao_data() -> HzDataset:
    """
    Retorna medidas de H(z) derivadas de BAO.

    Estas son medidas de H(z)*r_d convertidas usando r_d fiducial.
    """
    r_d = 147.09  # Mpc (Planck 2018)

    points = [
        # BOSS DR12 - H(z)*r_d convertido
        HzDataPoint(0.38, 81.2, 2.4, 'BAO', 'BOSS DR12'),
        HzDataPoint(0.51, 90.9, 2.3, 'BAO', 'BOSS DR12'),
        HzDataPoint(0.61, 97.8, 2.1, 'BAO', 'BOSS DR12'),
        # eBOSS
        HzDataPoint(0.70, 98.7, 2.1, 'BAO', 'eBOSS LRG'),
        HzDataPoint(1.48, 159.2, 4.5, 'BAO', 'eBOSS QSO'),
        HzDataPoint(2.33, 224.0, 7.0, 'BAO', 'eBOSS Lya'),
    ]
    return HzDataset(points=points, name='H(z) from BAO')


def get_combined_Hz_data(include_bao: bool = False) -> HzDataset:
    """
    Retorna dataset combinado de H(z).

    Args:
        include_bao: Si incluir medidas derivadas de BAO

    Returns:
        Dataset combinado
    """
    cc = get_cosmic_chronometers_data()

    if include_bao:
        bao = get_Hz_from_bao_data()
        all_points = cc.points + bao.points
        return HzDataset(points=all_points, name='Combined H(z)')

    return cc


# Función H(z) para ΛCDM de referencia
def H_lcdm(z: float, H0: float = 67.4, Omega_m: float = 0.3) -> float:
    """H(z) para ΛCDM plano."""
    Omega_L = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)


def compare_with_lcdm(
    dataset: HzDataset,
    H0: float = 67.4,
    Omega_m: float = 0.3
) -> Dict[str, float]:
    """
    Compara dataset con ΛCDM.

    Args:
        dataset: Datos de H(z)
        H0: Constante de Hubble
        Omega_m: Fracción de materia

    Returns:
        Estadísticos de comparación
    """
    z = dataset.z_values
    H_model = np.array([H_lcdm(zi, H0, Omega_m) for zi in z])
    H_data = dataset.H_values
    sigma = dataset.errors

    chi2 = chi2_Hz(H_model, H_data, sigma)
    n = len(z)

    return {
        'chi2': chi2,
        'chi2_reduced': chi2 / n,
        'n_dof': n - 2,  # 2 parámetros: H0, Omega_m
    }


# Exportaciones
__all__ = [
    'HzDataPoint',
    'HzDataset',
    'chi2_Hz',
    'evaluate_Hz_model',
    'get_cosmic_chronometers_data',
    'get_Hz_from_bao_data',
    'get_combined_Hz_data',
    'H_lcdm',
    'compare_with_lcdm',
]
