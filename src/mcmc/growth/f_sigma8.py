"""Predicción de fσ₈(z) para comparación con RSD.

fσ₈(z) = f(z) · σ₈(z) = f(z) · σ₈,₀ · D(z)

donde:
    f(z) = d ln D / d ln a ≈ Ω_m(z)^γ
    D(z) = factor de crecimiento normalizado D(0)=1
    σ₈,₀ = amplitud de fluctuaciones en 8 h⁻¹Mpc a z=0

Este observable es clave para tests de gravedad y energía oscura
usando Redshift Space Distortions (RSD).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np

from .linear_growth import (
    GrowthParams,
    LinearGrowthSolver,
    D_of_z_LCDM,
    f_of_z_LCDM,
)


@dataclass
class FSigma8Data:
    """Datos observacionales de fσ₈.

    Attributes:
        z: Redshifts
        f_sigma8: Valores medidos
        sigma: Incertidumbres
        survey: Nombre del survey
    """
    z: np.ndarray
    f_sigma8: np.ndarray
    sigma: np.ndarray
    survey: str = ""


class FSigma8Predictor:
    """Predictor de fσ₈(z) para el MCMC.

    Attributes:
        growth_solver: Solver del crecimiento lineal
        sigma8_0: Valor de σ₈ a z=0
    """

    def __init__(
        self,
        growth_solver: LinearGrowthSolver | None = None,
        sigma8_0: float = 0.811,
        params: GrowthParams | None = None
    ):
        """Inicializa el predictor.

        Args:
            growth_solver: Solver de crecimiento (crea uno si None)
            sigma8_0: σ₈(z=0)
            params: Parámetros de crecimiento
        """
        if growth_solver is None:
            self.growth_solver = LinearGrowthSolver(params)
        else:
            self.growth_solver = growth_solver

        self.sigma8_0 = sigma8_0
        self._D_cache = None
        self._z_cache = None

    def _ensure_D_cache(self, z_max: float = 3.0, n_points: int = 500):
        """Calcula y cachea D(z) si necesario."""
        if self._D_cache is None or self._z_cache is None:
            self._z_cache, self._D_cache = self.growth_solver.solve_growth_z(
                (0, z_max), n_points
            )

    def D_of_z(self, z: np.ndarray) -> np.ndarray:
        """Factor de crecimiento D(z).

        Args:
            z: Redshifts

        Returns:
            D(z) normalizado a D(0)=1
        """
        z_arr = np.asarray(z)
        z_max = float(np.max(z_arr)) * 1.2
        self._ensure_D_cache(z_max)

        return np.interp(z_arr, self._z_cache, self._D_cache)

    def f_of_z(self, z: np.ndarray) -> np.ndarray:
        """Tasa de crecimiento f(z) = d ln D / d ln a.

        Args:
            z: Redshifts

        Returns:
            f(z)
        """
        return self.growth_solver.f_of_z(np.asarray(z))

    def sigma8_of_z(self, z: np.ndarray) -> np.ndarray:
        """σ₈(z) = σ₈,₀ · D(z).

        Args:
            z: Redshifts

        Returns:
            σ₈(z)
        """
        D = self.D_of_z(z)
        return self.sigma8_0 * D

    def f_sigma8(self, z: np.ndarray) -> np.ndarray:
        """fσ₈(z) = f(z) · σ₈(z).

        Args:
            z: Redshifts

        Returns:
            fσ₈(z)
        """
        f = self.f_of_z(z)
        sigma8 = self.sigma8_of_z(z)
        return f * sigma8

    def chi2(self, data: FSigma8Data) -> float:
        """Calcula χ² respecto a datos observacionales.

        χ² = Σ [(fσ₈_obs - fσ₈_model)² / σ²]

        Args:
            data: Datos observacionales

        Returns:
            Valor de χ²
        """
        model = self.f_sigma8(data.z)
        residuals = data.f_sigma8 - model
        sigma_safe = np.maximum(data.sigma, 1e-10)
        return float(np.sum((residuals / sigma_safe)**2))


def compute_f_sigma8(
    z: np.ndarray,
    params: GrowthParams | None = None,
    G_eff: Callable[[float], float] | None = None
) -> np.ndarray:
    """Calcula fσ₈(z) con parámetros dados.

    Args:
        z: Redshifts
        params: Parámetros de crecimiento
        G_eff: Gravedad efectiva G(S) opcional

    Returns:
        fσ₈(z)
    """
    params = params or GrowthParams()
    solver = LinearGrowthSolver(params, G_eff=G_eff)
    predictor = FSigma8Predictor(solver, params.sigma8_0)
    return predictor.f_sigma8(z)


def f_sigma8_LCDM(
    z: np.ndarray,
    Omega_m0: float = 0.315,
    sigma8_0: float = 0.811,
    gamma: float = 0.55
) -> np.ndarray:
    """fσ₈(z) de referencia para ΛCDM.

    Args:
        z: Redshifts
        Omega_m0: Fracción de materia
        sigma8_0: σ₈(z=0)
        gamma: Exponente de crecimiento

    Returns:
        fσ₈(z) ΛCDM
    """
    D = D_of_z_LCDM(z, Omega_m0)
    f = f_of_z_LCDM(z, Omega_m0, gamma)
    sigma8 = sigma8_0 * D
    return f * sigma8


def S8_parameter(sigma8_0: float, Omega_m0: float, alpha: float = 0.5) -> float:
    """Calcula el parámetro S₈ = σ₈ (Ω_m/0.3)^α.

    Args:
        sigma8_0: σ₈(z=0)
        Omega_m0: Fracción de materia
        alpha: Exponente (típico 0.5)

    Returns:
        Parámetro S₈
    """
    return sigma8_0 * (Omega_m0 / 0.3) ** alpha


# =============================================================================
# Datos observacionales de referencia
# =============================================================================

def get_BOSS_data() -> FSigma8Data:
    """Datos de fσ₈ de BOSS DR12.

    Returns:
        FSigma8Data con medidas de BOSS
    """
    # Valores aproximados de BOSS DR12 (Alam et al. 2017)
    z = np.array([0.38, 0.51, 0.61])
    f_sigma8 = np.array([0.497, 0.458, 0.436])
    sigma = np.array([0.045, 0.038, 0.034])

    return FSigma8Data(z=z, f_sigma8=f_sigma8, sigma=sigma, survey="BOSS DR12")


def get_6dFGS_data() -> FSigma8Data:
    """Datos de fσ₈ de 6dFGS.

    Returns:
        FSigma8Data con medida de 6dFGS
    """
    # Beutler et al. 2012
    z = np.array([0.067])
    f_sigma8 = np.array([0.423])
    sigma = np.array([0.055])

    return FSigma8Data(z=z, f_sigma8=f_sigma8, sigma=sigma, survey="6dFGS")


def get_combined_RSD_data() -> FSigma8Data:
    """Combinación de datos RSD de múltiples surveys.

    Returns:
        FSigma8Data combinados
    """
    boss = get_BOSS_data()
    sixdf = get_6dFGS_data()

    z = np.concatenate([sixdf.z, boss.z])
    f_sigma8 = np.concatenate([sixdf.f_sigma8, boss.f_sigma8])
    sigma = np.concatenate([sixdf.sigma, boss.sigma])

    return FSigma8Data(z=z, f_sigma8=f_sigma8, sigma=sigma, survey="Combined RSD")


# =============================================================================
# Comparación MCMC vs ΛCDM
# =============================================================================

@dataclass
class GrowthComparison:
    """Resultado de comparación de crecimiento.

    Attributes:
        z: Redshifts
        f_sigma8_mcmc: fσ₈(z) del MCMC
        f_sigma8_lcdm: fσ₈(z) de ΛCDM
        ratio: fσ₈_MCMC / fσ₈_ΛCDM
        S8_mcmc: Parámetro S₈ del MCMC
        S8_lcdm: Parámetro S₈ de ΛCDM
    """
    z: np.ndarray
    f_sigma8_mcmc: np.ndarray
    f_sigma8_lcdm: np.ndarray
    ratio: np.ndarray
    S8_mcmc: float
    S8_lcdm: float


def compare_growth(
    params_mcmc: GrowthParams,
    G_eff: Callable[[float], float] | None = None,
    z_range: Tuple[float, float] = (0, 2),
    n_points: int = 100
) -> GrowthComparison:
    """Compara crecimiento MCMC vs ΛCDM.

    Args:
        params_mcmc: Parámetros del MCMC
        G_eff: Gravedad efectiva
        z_range: Rango de redshifts
        n_points: Número de puntos

    Returns:
        GrowthComparison
    """
    z = np.linspace(z_range[0], z_range[1], n_points)

    # MCMC
    f_sigma8_mcmc = compute_f_sigma8(z, params_mcmc, G_eff)
    S8_mcmc = S8_parameter(params_mcmc.sigma8_0, params_mcmc.Omega_m0)

    # ΛCDM
    f_sigma8_lcdm = f_sigma8_LCDM(z, params_mcmc.Omega_m0, params_mcmc.sigma8_0)
    S8_lcdm = S8_parameter(params_mcmc.sigma8_0, params_mcmc.Omega_m0)

    # Ratio
    ratio = f_sigma8_mcmc / np.maximum(f_sigma8_lcdm, 1e-30)

    return GrowthComparison(
        z=z,
        f_sigma8_mcmc=f_sigma8_mcmc,
        f_sigma8_lcdm=f_sigma8_lcdm,
        ratio=ratio,
        S8_mcmc=S8_mcmc,
        S8_lcdm=S8_lcdm,
    )
