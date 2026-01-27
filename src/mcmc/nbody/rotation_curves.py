"""Curvas de rotación y validación con catálogo SPARC.

Este módulo implementa el cálculo de curvas de rotación teóricas
y su comparación con observaciones. La validación local del
módulo N-body se realiza comparando con el catálogo SPARC.

Predicciones del MCMC:
    - Núcleos planos (cored) en lugar de cúspides NFW
    - Parámetros de halo dependientes de S_loc
    - Mejor ajuste a galaxias de baja masa superficial

Métricas de comparación:
    - χ² entre modelo y observaciones
    - AIC/BIC para selección de modelo
    - Δχ² = χ²_NFW - χ²_MCMC
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .profiles import (
    BurkertParams, ZhaoParams, NFWParams, SlocParams,
    V_burkert, V_zhao, V_nfw,
    halo_core_params_from_Sloc,
)


@dataclass
class RotationCurveData:
    """Datos observacionales de curva de rotación.

    Attributes:
        r: Radios en kpc
        V_obs: Velocidad observada en km/s
        sigma: Incertidumbre en km/s
        V_disk: Contribución del disco (opcional)
        V_gas: Contribución del gas (opcional)
        V_bulge: Contribución del bulbo (opcional)
        name: Nombre de la galaxia
    """
    r: np.ndarray
    V_obs: np.ndarray
    sigma: np.ndarray
    V_disk: np.ndarray | None = None
    V_gas: np.ndarray | None = None
    V_bulge: np.ndarray | None = None
    name: str = ""


def V_total(
    r: np.ndarray,
    V_halo: np.ndarray,
    V_disk: np.ndarray | None = None,
    V_gas: np.ndarray | None = None,
    V_bulge: np.ndarray | None = None
) -> np.ndarray:
    """Velocidad circular total (suma en cuadratura).

    V_tot² = V_disk² + V_gas² + V_bulge² + V_halo²

    Args:
        r: Radios en kpc
        V_halo: Contribución del halo oscuro
        V_disk: Contribución del disco estelar
        V_gas: Contribución del gas
        V_bulge: Contribución del bulbo

    Returns:
        Velocidad total en km/s
    """
    V2 = V_halo ** 2

    if V_disk is not None:
        V2 = V2 + V_disk ** 2
    if V_gas is not None:
        V2 = V2 + V_gas ** 2
    if V_bulge is not None:
        V2 = V2 + V_bulge ** 2

    return np.sqrt(V2)


def chi2_rotation_curve(
    V_model: np.ndarray,
    data: RotationCurveData
) -> float:
    """Calcula χ² entre curva de rotación observada y modelo.

    χ² = Σ [(V_obs - V_model)² / σ²]

    Args:
        V_model: Velocidad del modelo en km/s
        data: Datos observacionales

    Returns:
        Valor de χ²
    """
    residuals = data.V_obs - V_model
    sigma_safe = np.maximum(data.sigma, 1e-10)
    return float(np.sum((residuals / sigma_safe) ** 2))


def chi2_burkert(
    params: BurkertParams,
    data: RotationCurveData
) -> float:
    """χ² para perfil Burkert.

    Args:
        params: Parámetros Burkert
        data: Datos observacionales

    Returns:
        Valor de χ²
    """
    V_halo = V_burkert(data.r, params)
    V_model = V_total(data.r, V_halo, data.V_disk, data.V_gas, data.V_bulge)
    return chi2_rotation_curve(V_model, data)


def chi2_zhao(
    params: ZhaoParams,
    data: RotationCurveData
) -> float:
    """χ² para perfil Zhao.

    Args:
        params: Parámetros Zhao
        data: Datos observacionales

    Returns:
        Valor de χ²
    """
    V_halo = V_zhao(data.r, params)
    V_model = V_total(data.r, V_halo, data.V_disk, data.V_gas, data.V_bulge)
    return chi2_rotation_curve(V_model, data)


def chi2_nfw(
    params: NFWParams,
    data: RotationCurveData
) -> float:
    """χ² para perfil NFW.

    Args:
        params: Parámetros NFW
        data: Datos observacionales

    Returns:
        Valor de χ²
    """
    V_halo = V_nfw(data.r, params)
    V_model = V_total(data.r, V_halo, data.V_disk, data.V_gas, data.V_bulge)
    return chi2_rotation_curve(V_model, data)


def chi2_mcmc_Sloc(
    S_loc: float,
    sloc_params: SlocParams,
    data: RotationCurveData,
    profile: str = "Burkert"
) -> float:
    """χ² para perfil MCMC con parámetros derivados de S_loc.

    Args:
        S_loc: Entropía local del halo
        sloc_params: Parámetros de referencia S_loc
        data: Datos observacionales
        profile: Tipo de perfil ("Burkert" o "Zhao")

    Returns:
        Valor de χ²
    """
    rho0, r_core = halo_core_params_from_Sloc(S_loc, sloc_params)

    if profile == "Burkert":
        params = BurkertParams(rho0=rho0, r0=r_core)
        return chi2_burkert(params, data)
    elif profile == "Zhao":
        params = ZhaoParams(rho_s=rho0, r_s=r_core, alpha=2.0, beta=3.0, gamma=0.0)
        return chi2_zhao(params, data)
    else:
        raise ValueError(f"Perfil no reconocido: {profile}")


# ---------------------------------------------------------------------
# Criterios de información para selección de modelo
# ---------------------------------------------------------------------

def aic(chi2: float, k: int) -> float:
    """Criterio de Información de Akaike.

    AIC = 2k + χ²_best

    Args:
        chi2: Mejor χ² del modelo
        k: Número de parámetros libres

    Returns:
        Valor AIC
    """
    return 2 * k + chi2


def bic(chi2: float, k: int, n: int) -> float:
    """Criterio de Información Bayesiano.

    BIC = k·ln(N) + χ²_best

    Args:
        chi2: Mejor χ² del modelo
        k: Número de parámetros libres
        n: Número de puntos de datos

    Returns:
        Valor BIC
    """
    return k * np.log(n) + chi2


@dataclass
class ModelComparison:
    """Resultado de comparación entre modelos.

    Attributes:
        chi2_mcmc: χ² del modelo MCMC (cored)
        chi2_nfw: χ² del modelo NFW (cuspy)
        delta_chi2: χ²_NFW - χ²_MCMC (positivo = MCMC mejor)
        delta_aic: AIC_NFW - AIC_MCMC
        delta_bic: BIC_NFW - BIC_MCMC
        n_data: Número de puntos de datos
        k_mcmc: Parámetros libres MCMC
        k_nfw: Parámetros libres NFW
    """
    chi2_mcmc: float
    chi2_nfw: float
    delta_chi2: float
    delta_aic: float
    delta_bic: float
    n_data: int
    k_mcmc: int = 2
    k_nfw: int = 2

    @property
    def mcmc_preferred(self) -> bool:
        """Indica si MCMC es preferido (Δχ² > 0)."""
        return self.delta_chi2 > 0

    @property
    def significance(self) -> str:
        """Significancia de la preferencia."""
        if self.delta_chi2 < 0:
            return "NFW preferred"
        elif self.delta_chi2 < 2.3:
            return "Inconclusive"
        elif self.delta_chi2 < 6.2:
            return "MCMC moderately preferred"
        else:
            return "MCMC strongly preferred"


def compare_models(
    data: RotationCurveData,
    burkert_params: BurkertParams,
    nfw_params: NFWParams
) -> ModelComparison:
    """Compara modelos Burkert (MCMC) vs NFW (ΛCDM).

    Args:
        data: Datos observacionales
        burkert_params: Parámetros del perfil Burkert
        nfw_params: Parámetros del perfil NFW

    Returns:
        Resultado de la comparación
    """
    chi2_b = chi2_burkert(burkert_params, data)
    chi2_n = chi2_nfw(nfw_params, data)

    n = len(data.r)
    k_mcmc = 2  # rho0, r0
    k_nfw = 2   # rho_s, r_s

    aic_b = aic(chi2_b, k_mcmc)
    aic_n = aic(chi2_n, k_nfw)

    bic_b = bic(chi2_b, k_mcmc, n)
    bic_n = bic(chi2_n, k_nfw, n)

    return ModelComparison(
        chi2_mcmc=chi2_b,
        chi2_nfw=chi2_n,
        delta_chi2=chi2_n - chi2_b,
        delta_aic=aic_n - aic_b,
        delta_bic=bic_n - bic_b,
        n_data=n,
        k_mcmc=k_mcmc,
        k_nfw=k_nfw,
    )


# ---------------------------------------------------------------------
# Optimización de parámetros
# ---------------------------------------------------------------------

def fit_burkert(
    data: RotationCurveData,
    rho0_init: float = 1e7,
    r0_init: float = 2.0,
    bounds: tuple | None = None
) -> tuple[BurkertParams, float]:
    """Ajusta perfil Burkert a datos de curva de rotación.

    Args:
        data: Datos observacionales
        rho0_init: Valor inicial de ρ₀
        r0_init: Valor inicial de r₀
        bounds: Límites [(rho0_min, rho0_max), (r0_min, r0_max)]

    Returns:
        (params_best, chi2_best): Parámetros óptimos y χ² mínimo
    """
    from scipy.optimize import minimize

    if bounds is None:
        bounds = [(1e4, 1e10), (0.1, 50.0)]

    def objective(x):
        params = BurkertParams(rho0=x[0], r0=x[1])
        return chi2_burkert(params, data)

    x0 = [rho0_init, r0_init]
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    best_params = BurkertParams(rho0=result.x[0], r0=result.x[1])
    return best_params, float(result.fun)


def fit_nfw(
    data: RotationCurveData,
    rho_s_init: float = 1e7,
    r_s_init: float = 10.0,
    bounds: tuple | None = None
) -> tuple[NFWParams, float]:
    """Ajusta perfil NFW a datos de curva de rotación.

    Args:
        data: Datos observacionales
        rho_s_init: Valor inicial de ρ_s
        r_s_init: Valor inicial de r_s
        bounds: Límites [(rho_s_min, rho_s_max), (r_s_min, r_s_max)]

    Returns:
        (params_best, chi2_best): Parámetros óptimos y χ² mínimo
    """
    from scipy.optimize import minimize

    if bounds is None:
        bounds = [(1e4, 1e10), (1.0, 100.0)]

    def objective(x):
        params = NFWParams(rho_s=x[0], r_s=x[1])
        return chi2_nfw(params, data)

    x0 = [rho_s_init, r_s_init]
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    best_params = NFWParams(rho_s=result.x[0], r_s=result.x[1])
    return best_params, float(result.fun)


def fit_and_compare(data: RotationCurveData) -> ModelComparison:
    """Ajusta ambos modelos y los compara.

    Args:
        data: Datos observacionales

    Returns:
        Resultado de la comparación con mejores ajustes
    """
    burkert_best, chi2_b = fit_burkert(data)
    nfw_best, chi2_n = fit_nfw(data)

    n = len(data.r)

    return ModelComparison(
        chi2_mcmc=chi2_b,
        chi2_nfw=chi2_n,
        delta_chi2=chi2_n - chi2_b,
        delta_aic=aic(chi2_n, 2) - aic(chi2_b, 2),
        delta_bic=bic(chi2_n, 2, n) - bic(chi2_b, 2, n),
        n_data=n,
    )
