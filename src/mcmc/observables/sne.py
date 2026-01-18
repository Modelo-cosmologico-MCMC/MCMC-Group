"""
Observables SNe Ia (Supernovas Tipo Ia)
=======================================

Este módulo implementa el manejo de datos de supernovas tipo Ia
y el cálculo del likelihood correspondiente.

Los SNe Ia son "candelas estándar" que proporcionan medidas de la
distancia de luminosidad d_L(z) a través del módulo de distancia:
    μ = m - M = 5 log10(d_L / 10 pc)

Datasets soportados:
- Pantheon (2018): 1048 SNe
- Pantheon+ (2022): 1701 SNe
- Union2.1

Referencias:
    - Documento de Simulaciones Conjuntas: SNe likelihood
    - Scolnic et al. (2018), Brout et al. (2022)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


@dataclass
class SNDataPoint:
    """
    Punto de datos de supernova.

    Attributes:
        z_cmb: Redshift en el marco CMB
        z_hel: Redshift heliocéntrico
        mu: Módulo de distancia observado
        sigma_mu: Error en μ
        name: Nombre/identificador del SN
    """
    z_cmb: float
    z_hel: float
    mu: float
    sigma_mu: float
    name: str = ''


@dataclass
class SNDataset:
    """
    Dataset de supernovas tipo Ia.
    """
    points: List[SNDataPoint]
    name: str = 'SNe Ia'
    cov_matrix: Optional[np.ndarray] = None  # Matriz de covarianza completa
    has_systematics: bool = False

    @property
    def n_sne(self) -> int:
        return len(self.points)

    @property
    def z_values(self) -> np.ndarray:
        return np.array([p.z_cmb for p in self.points])

    @property
    def mu_values(self) -> np.ndarray:
        return np.array([p.mu for p in self.points])

    @property
    def errors(self) -> np.ndarray:
        return np.array([p.sigma_mu for p in self.points])


def chi2_sne_simple(
    mu_model: np.ndarray,
    mu_data: np.ndarray,
    sigma_mu: np.ndarray
) -> float:
    """
    Calcula χ² simple para SNe (sin covarianza).

    χ² = Σ [(μ_model - μ_data) / σ_μ]²

    Args:
        mu_model: Módulo de distancia del modelo
        mu_data: Módulo de distancia observado
        sigma_mu: Errores

    Returns:
        χ²
    """
    residuals = (mu_model - mu_data) / sigma_mu
    return np.sum(residuals**2)


def chi2_sne_with_cov(
    mu_model: np.ndarray,
    mu_data: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calcula χ² con matriz de covarianza completa.

    χ² = Δμᵀ C⁻¹ Δμ

    Args:
        mu_model: Módulo de distancia del modelo
        mu_data: Módulo de distancia observado
        cov_matrix: Matriz de covarianza

    Returns:
        χ²
    """
    delta_mu = mu_model - mu_data
    cov_inv = np.linalg.inv(cov_matrix)
    return float(delta_mu @ cov_inv @ delta_mu)


def chi2_sne_marginalized(
    mu_model: np.ndarray,
    mu_data: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """
    Calcula χ² marginalizado sobre M (magnitud absoluta).

    Esto es estándar en análisis de SNe donde M es un parámetro de "nuisance".
    La marginalización analítica resulta en:

    χ² = A - B²/C

    donde:
    A = Δμᵀ C⁻¹ Δμ
    B = Σᵢⱼ C⁻¹ᵢⱼ Δμⱼ = 1ᵀ C⁻¹ Δμ
    C = Σᵢⱼ C⁻¹ᵢⱼ = 1ᵀ C⁻¹ 1

    Args:
        mu_model: Módulo de distancia del modelo
        mu_data: Módulo de distancia observado
        cov_matrix: Matriz de covarianza

    Returns:
        χ² marginalizado
    """
    delta_mu = mu_model - mu_data
    cov_inv = np.linalg.inv(cov_matrix)
    ones = np.ones(len(delta_mu))

    A = float(delta_mu @ cov_inv @ delta_mu)
    B = float(ones @ cov_inv @ delta_mu)
    C = float(ones @ cov_inv @ ones)

    return A - B**2 / C


def mu_from_dL(d_L: np.ndarray) -> np.ndarray:
    """
    Calcula el módulo de distancia desde d_L.

    μ = 5 log10(d_L / 10 pc) = 5 log10(d_L [Mpc]) + 25

    Args:
        d_L: Distancia de luminosidad [Mpc]

    Returns:
        Módulo de distancia
    """
    return 5.0 * np.log10(d_L) + 25.0


def evaluate_sne_model(
    dist_calc,
    dataset: SNDataset,
    marginalize: bool = True
) -> Dict[str, float]:
    """
    Evalúa un modelo contra datos de SNe.

    Args:
        dist_calc: Calculador de distancias con método d_L(z)
        dataset: Dataset de SNe
        marginalize: Si marginalizar sobre M

    Returns:
        Diccionario con χ² y otros estadísticos
    """
    z = dataset.z_values
    mu_model = np.array([mu_from_dL(dist_calc.d_L(zi)) for zi in z])
    mu_data = dataset.mu_values

    if dataset.cov_matrix is not None:
        if marginalize:
            chi2 = chi2_sne_marginalized(mu_model, mu_data, dataset.cov_matrix)
        else:
            chi2 = chi2_sne_with_cov(mu_model, mu_data, dataset.cov_matrix)
    else:
        chi2 = chi2_sne_simple(mu_model, mu_data, dataset.errors)

    return {
        'chi2': chi2,
        'n_sne': len(z),
        'chi2_reduced': chi2 / len(z),
    }


# Datos simulados para pruebas (sin acceso a archivos reales)
def get_pantheon_simulated_data(n_sne: int = 100) -> SNDataset:
    """
    Genera datos simulados tipo Pantheon para pruebas.

    Los datos simulados siguen un modelo ΛCDM con ruido gaussiano.

    Args:
        n_sne: Número de SNe a simular

    Returns:
        Dataset simulado
    """
    # Parámetros ΛCDM fiduciales
    H0 = 70.0  # km/s/Mpc
    Omega_m = 0.3
    Omega_L = 0.7

    # Distribución de z similar a Pantheon
    np.random.seed(42)
    z = np.sort(np.random.uniform(0.01, 2.3, n_sne))

    # Calcular μ teórico
    from .distances import C_LIGHT

    def E(z_val):
        return np.sqrt(Omega_m * (1 + z_val)**3 + Omega_L)

    # Integrar d_L
    from scipy.integrate import quad
    d_L = np.zeros(n_sne)
    for i, zi in enumerate(z):
        integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
        d_L[i] = (1 + zi) * C_LIGHT / H0 * integral

    mu_true = 5.0 * np.log10(d_L) + 25.0

    # Añadir errores
    sigma_mu = 0.1 + 0.05 * z  # Error que crece con z
    mu_obs = mu_true + np.random.normal(0, sigma_mu)

    points = [
        SNDataPoint(
            z_cmb=z[i],
            z_hel=z[i],  # Simplificación
            mu=mu_obs[i],
            sigma_mu=sigma_mu[i],
            name=f'SN{i:04d}'
        )
        for i in range(n_sne)
    ]

    # Matriz de covarianza diagonal
    cov = np.diag(sigma_mu**2)

    return SNDataset(
        points=points,
        name='Pantheon Simulated',
        cov_matrix=cov,
        has_systematics=False
    )


def get_pantheon_binned_data() -> SNDataset:
    """
    Retorna datos Pantheon binneados (aproximados).

    Valores representativos para pruebas rápidas.
    """
    # Datos binneados aproximados de Pantheon
    z_bins = np.array([
        0.023, 0.034, 0.047, 0.062, 0.080, 0.101, 0.128, 0.162,
        0.205, 0.260, 0.329, 0.416, 0.527, 0.667, 0.845, 1.070
    ])

    mu_bins = np.array([
        33.21, 34.51, 35.57, 36.41, 37.18, 37.89, 38.52, 39.13,
        39.72, 40.29, 40.86, 41.41, 41.94, 42.47, 43.00, 43.54
    ])

    sigma_bins = np.array([
        0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.05,
        0.05, 0.06, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15
    ])

    points = [
        SNDataPoint(
            z_cmb=z_bins[i],
            z_hel=z_bins[i],
            mu=mu_bins[i],
            sigma_mu=sigma_bins[i],
            name=f'Bin{i:02d}'
        )
        for i in range(len(z_bins))
    ]

    return SNDataset(points=points, name='Pantheon Binned')


def load_pantheon_data(data_path: str) -> SNDataset:
    """
    Carga datos Pantheon desde archivo.

    Args:
        data_path: Ruta al archivo de datos

    Returns:
        Dataset Pantheon

    Note:
        Requiere los archivos de datos de Pantheon.
        Disponibles en: https://github.com/dscolnic/Pantheon
    """
    raise NotImplementedError(
        "Para cargar datos reales de Pantheon, descargue los archivos desde "
        "https://github.com/dscolnic/Pantheon y proporcione la ruta."
    )


# Exportaciones
__all__ = [
    'SNDataPoint',
    'SNDataset',
    'chi2_sne_simple',
    'chi2_sne_with_cov',
    'chi2_sne_marginalized',
    'mu_from_dL',
    'evaluate_sne_model',
    'get_pantheon_simulated_data',
    'get_pantheon_binned_data',
    'load_pantheon_data',
]
