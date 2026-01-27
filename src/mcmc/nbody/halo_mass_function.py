"""Función de masa de halos para el MCMC.

La función de masa n(M, z) es uno de los diagnósticos más
importantes para comparar simulaciones Cronos vs ΛCDM.

Predicciones del MCMC:
    Régimen               | R_n = n_Cronos/n_ΛCDM
    ----------------------|----------------------
    M > 10¹⁴ M☉ (cúmulos) | R_n ≈ 1.0
    10¹¹-10¹³ M☉          | R_n ≈ 0.9-1.0
    M < 10¹⁰ M☉ (enanos)  | R_n < 0.8

La supresión a baja masa surge de la fricción entrópica
que ralentiza el colapso en regiones de baja densidad.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Callable

from .halo_finder import Halo


@dataclass
class HMFParams:
    """Parámetros para cálculo de función de masa.

    Attributes:
        L_box: Tamaño de caja en h⁻¹ Mpc
        M_min: Masa mínima en M_sun
        M_max: Masa máxima en M_sun
        n_bins: Número de bins en log(M)
    """
    L_box: float  # h^{-1} Mpc
    M_min: float = 1e8
    M_max: float = 1e16
    n_bins: int = 20


def halo_mass_function(
    halos: list[Halo],
    params: HMFParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula la función de masa de halos.

    n(M) ≡ dN / (dV · d ln M)

    Args:
        halos: Lista de halos identificados
        params: Parámetros del cálculo

    Returns:
        (M_centers, dn_dlnM, dn_dlnM_err):
            - M_centers: Centros de bins en M_sun
            - dn_dlnM: Función de masa en (h/Mpc)³
            - dn_dlnM_err: Error Poisson
    """
    # Extraer masas
    masses = np.array([h.mass for h in halos])
    masses = masses[masses > 0]

    if len(masses) == 0:
        M_centers = 10 ** np.linspace(
            np.log10(params.M_min),
            np.log10(params.M_max),
            params.n_bins
        )
        return M_centers, np.zeros(params.n_bins - 1), np.zeros(params.n_bins - 1)

    # Bins en log(M)
    log_M_bins = np.linspace(
        np.log10(params.M_min),
        np.log10(params.M_max),
        params.n_bins + 1
    )

    # Volumen de la caja
    V_box = params.L_box ** 3  # (h^{-1} Mpc)³

    # Histograma en log(M)
    counts, _ = np.histogram(np.log10(masses), bins=log_M_bins)

    # d ln M = d log10(M) * ln(10)
    d_lnM = np.diff(log_M_bins) * np.log(10)

    # dn / d ln M
    dn_dlnM = counts / (V_box * d_lnM)

    # Error Poisson: sqrt(N) / (V * d ln M)
    dn_dlnM_err = np.sqrt(np.maximum(counts, 1)) / (V_box * d_lnM)

    # Centros de bins
    M_centers = 10 ** (0.5 * (log_M_bins[:-1] + log_M_bins[1:]))

    return M_centers, dn_dlnM, dn_dlnM_err


def cumulative_mass_function(
    halos: list[Halo],
    params: HMFParams
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula la función de masa acumulada N(>M).

    N(>M) = número de halos con masa mayor que M por unidad de volumen

    Args:
        halos: Lista de halos
        params: Parámetros del cálculo

    Returns:
        (M_array, N_cumulative): Masa y conteo acumulado
    """
    masses = np.array([h.mass for h in halos])
    masses = masses[masses > 0]
    masses = np.sort(masses)[::-1]  # Ordenar descendente

    V_box = params.L_box ** 3

    # Array de masas para evaluación
    M_array = 10 ** np.linspace(
        np.log10(params.M_min),
        np.log10(params.M_max),
        100
    )

    # Conteo acumulado
    N_cumulative = np.array([np.sum(masses > M) for M in M_array]) / V_box

    return M_array, N_cumulative


# ---------------------------------------------------------------------
# Funciones de masa teóricas para comparación
# ---------------------------------------------------------------------

def press_schechter(
    M: np.ndarray,
    sigma_M: Callable[[np.ndarray], np.ndarray],
    dsigma_dM: Callable[[np.ndarray], np.ndarray],
    rho_m: float,
    delta_c: float = 1.686
) -> np.ndarray:
    """Función de masa Press-Schechter (referencia).

    n(M) = sqrt(2/π) (ρ_m/M²) |d ln σ/d ln M| (δ_c/σ) exp(-δ_c²/2σ²)

    Args:
        M: Array de masas
        sigma_M: Función σ(M) de varianza de la materia
        dsigma_dM: Derivada dσ/dM
        rho_m: Densidad de materia
        delta_c: Umbral de colapso

    Returns:
        dn/d ln M
    """
    sigma = sigma_M(M)
    dsig = dsigma_dM(M)

    # |d ln σ / d ln M| = |M/σ · dσ/dM|
    dlnsig_dlnM = np.abs(M * dsig / sigma)

    prefactor = np.sqrt(2.0 / np.pi) * (rho_m / M**2)
    nu = delta_c / sigma

    return prefactor * dlnsig_dlnM * nu * np.exp(-0.5 * nu**2)


def sheth_tormen(
    M: np.ndarray,
    sigma_M: Callable[[np.ndarray], np.ndarray],
    dsigma_dM: Callable[[np.ndarray], np.ndarray],
    rho_m: float,
    delta_c: float = 1.686,
    a: float = 0.707,
    p: float = 0.3,
    A: float = 0.322
) -> np.ndarray:
    """Función de masa Sheth-Tormen (más precisa que PS).

    Args:
        M: Array de masas
        sigma_M: Función σ(M)
        dsigma_dM: Derivada dσ/dM
        rho_m: Densidad de materia
        delta_c: Umbral de colapso
        a, p, A: Parámetros de calibración

    Returns:
        dn/d ln M
    """
    sigma = sigma_M(M)
    dsig = dsigma_dM(M)

    dlnsig_dlnM = np.abs(M * dsig / sigma)

    nu = delta_c / sigma
    nu_prime = np.sqrt(a) * nu

    prefactor = A * (rho_m / M**2) * dlnsig_dlnM

    f_nu = (
        (1.0 + 1.0 / nu_prime**(2*p))
        * nu_prime
        * np.exp(-0.5 * nu_prime**2)
        / np.sqrt(np.pi)
    )

    return prefactor * f_nu


# ---------------------------------------------------------------------
# Comparación Cronos vs ΛCDM
# ---------------------------------------------------------------------

@dataclass
class HMFComparison:
    """Resultado de comparación de funciones de masa.

    Attributes:
        M_centers: Centros de bins en M_sun
        dn_cronos: Función de masa Cronos
        dn_lcdm: Función de masa ΛCDM (o teórica)
        R_n: Ratio n_Cronos / n_ΛCDM
        R_n_err: Error en R_n
    """
    M_centers: np.ndarray
    dn_cronos: np.ndarray
    dn_lcdm: np.ndarray
    R_n: np.ndarray
    R_n_err: np.ndarray


def compare_hmf(
    halos_cronos: list[Halo],
    halos_lcdm: list[Halo],
    params: HMFParams
) -> HMFComparison:
    """Compara funciones de masa Cronos vs ΛCDM.

    Args:
        halos_cronos: Halos de simulación Cronos
        halos_lcdm: Halos de simulación ΛCDM (gemelo)
        params: Parámetros del cálculo

    Returns:
        Comparación de funciones de masa
    """
    M, dn_c, err_c = halo_mass_function(halos_cronos, params)
    _, dn_l, err_l = halo_mass_function(halos_lcdm, params)

    # Ratio R_n = n_Cronos / n_ΛCDM
    dn_l_safe = np.maximum(dn_l, 1e-30)
    R_n = dn_c / dn_l_safe

    # Error propagado
    R_n_err = R_n * np.sqrt(
        (err_c / np.maximum(dn_c, 1e-30))**2
        + (err_l / dn_l_safe)**2
    )

    # Donde no hay datos, poner NaN
    mask = (dn_c == 0) | (dn_l == 0)
    R_n[mask] = np.nan
    R_n_err[mask] = np.nan

    return HMFComparison(
        M_centers=M,
        dn_cronos=dn_c,
        dn_lcdm=dn_l,
        R_n=R_n,
        R_n_err=R_n_err,
    )


def subhalo_mass_function(
    halos: list[Halo],
    parent_mass_range: tuple[float, float],
    params: HMFParams
) -> tuple[np.ndarray, np.ndarray]:
    """Función de masa de subhalos.

    Calcula n(M_sub | M_parent) para halos padre en un rango de masa.

    Nota: Requiere que los halos tengan información de parentesco
    (no implementado en el finder básico).

    Args:
        halos: Lista de halos (incluye subhalos)
        parent_mass_range: (M_min, M_max) del halo padre
        params: Parámetros del cálculo

    Returns:
        (M_centers, dn_dlnM_sub): Función de masa de subhalos
    """
    # Placeholder - requiere información de subhalos
    # En una implementación completa, se filtrarían subhalos
    # por masa del padre
    M_centers = 10 ** np.linspace(
        np.log10(params.M_min),
        np.log10(params.M_max),
        params.n_bins
    )
    dn_dlnM_sub = np.zeros(params.n_bins - 1)

    return M_centers[:-1], dn_dlnM_sub


# ---------------------------------------------------------------------
# Predicciones MCMC
# ---------------------------------------------------------------------

def R_n_mcmc_prediction(
    M: np.ndarray,
    zeta_0: float = 0.02,
    M_transition: float = 1e11,
    suppression_slope: float = 0.15
) -> np.ndarray:
    """Predicción teórica del ratio R_n para el MCMC.

    Modelo fenomenológico basado en fricción entrópica:
    - M > M_trans: R_n ≈ 1
    - M < M_trans: R_n ≈ 1 - suppression_slope * log10(M_trans/M)

    Args:
        M: Array de masas en M_sun
        zeta_0: Fuerza de fricción entrópica
        M_transition: Masa de transición
        suppression_slope: Pendiente de supresión

    Returns:
        R_n predicho
    """
    log_ratio = np.log10(M_transition / M)
    log_ratio = np.maximum(log_ratio, 0)  # Solo supresión para M < M_trans

    # Supresión proporcional a zeta_0 y log(M_trans/M)
    suppression = zeta_0 * suppression_slope * log_ratio / 0.02

    R_n = 1.0 - suppression
    return np.maximum(R_n, 0.5)  # Piso para evitar supresión extrema
