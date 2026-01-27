"""Perfiles de densidad de halos para el MCMC.

El MCMC predice perfiles con núcleo (cored) en lugar de cúspides (cuspy).
La fricción entrópica y la dilatación temporal generan redistribución
de masa hacia radios mayores.

Perfiles implementados:
    - Burkert: Perfil clásico con núcleo plano
    - Zhao: Familia generalizada con parámetros (α, β, γ)
    - NFW: Perfil de referencia ΛCDM (para comparación)

Dependencia Entrópica S_loc:
    ρ₀(S_loc) = ρ_★ · (S_★ / S_loc)^α_ρ
    r_core(S_loc) = r_★ · (S_loc / S_★)^α_r
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad


# Constante gravitacional en unidades kpc (km/s)^2 / M_sun
G_KPC = 4.302e-6


@dataclass
class BurkertParams:
    """Parámetros del perfil Burkert.

    Attributes:
        rho0: Densidad central en M_sun/kpc³
        r0: Radio de núcleo en kpc
    """
    rho0: float  # M_sun/kpc³
    r0: float    # kpc


@dataclass
class ZhaoParams:
    """Parámetros del perfil Zhao generalizado.

    ρ(r) = ρ_s / [(r/r_s)^γ · (1 + (r/r_s)^α)^((β-γ)/α)]

    Casos especiales:
        - NFW: (α, β, γ) = (1, 3, 1)
        - Hernquist: (α, β, γ) = (1, 4, 1)
        - Cored: (α, β, γ) = (2, 3, 0)

    Attributes:
        rho_s: Densidad característica en M_sun/kpc³
        r_s: Radio de escala en kpc
        alpha: Parámetro de transición (típico 1-2)
        beta: Pendiente externa (típico 3-4)
        gamma: Pendiente interna (0=cored, 1=cuspy)
    """
    rho_s: float  # M_sun/kpc³
    r_s: float    # kpc
    alpha: float = 2.0
    beta: float = 3.0
    gamma: float = 0.0  # Cored por defecto


@dataclass
class NFWParams:
    """Parámetros del perfil NFW (referencia ΛCDM).

    Attributes:
        rho_s: Densidad característica en M_sun/kpc³
        r_s: Radio de escala en kpc
    """
    rho_s: float  # M_sun/kpc³
    r_s: float    # kpc


@dataclass
class SlocParams:
    """Parámetros de dependencia entrópica S_loc.

    El MCMC predice que los parámetros de halo dependen del
    estado entrópico local (historia de conversión Mp → Ep).

    Attributes:
        S_star: Entropía de referencia (típico S_BB = 1.001)
        rho_star: Densidad de referencia en M_sun/kpc³
        r_star: Radio de referencia en kpc
        alpha_rho: Exponente para densidad (típico 0.3-0.5)
        alpha_r: Exponente para radio (típico 0.2-0.4)
    """
    S_star: float = 1.001
    rho_star: float = 1e7  # M_sun/kpc³
    r_star: float = 2.0    # kpc
    alpha_rho: float = 0.4
    alpha_r: float = 0.3


# ---------------------------------------------------------------------
# Perfil de Burkert
# ---------------------------------------------------------------------

def rho_burkert(r: np.ndarray, p: BurkertParams) -> np.ndarray:
    """Densidad del perfil Burkert.

    ρ(r) = ρ₀ r₀³ / [(r + r₀)(r² + r₀²)]

    Args:
        r: Radio en kpc
        p: Parámetros Burkert

    Returns:
        Densidad en M_sun/kpc³
    """
    r = np.asarray(r, dtype=float)
    r0 = max(p.r0, 1e-10)
    return p.rho0 * r0**3 / ((r + r0) * (r**2 + r0**2))


def mass_burkert(r: np.ndarray, p: BurkertParams) -> np.ndarray:
    """Masa encerrada del perfil Burkert (forma analítica).

    M(r) = 2π ρ₀ r₀³ [ln(1 + r/r₀) + ½ ln(1 + r²/r₀²) - arctan(r/r₀)]

    Args:
        r: Radio en kpc
        p: Parámetros Burkert

    Returns:
        Masa encerrada en M_sun
    """
    r = np.asarray(r, dtype=float)
    x = r / max(p.r0, 1e-10)

    term1 = np.log(1.0 + x)
    term2 = 0.5 * np.log(1.0 + x**2)
    term3 = np.arctan(x)

    return 2.0 * np.pi * p.rho0 * p.r0**3 * (term1 + term2 - term3)


def V_burkert(r: np.ndarray, p: BurkertParams) -> np.ndarray:
    """Velocidad circular para halo Burkert.

    V(r) = sqrt(G · M(r) / r)

    Args:
        r: Radio en kpc
        p: Parámetros Burkert

    Returns:
        Velocidad circular en km/s
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.maximum(r, 1e-10)
    M = mass_burkert(r, p)
    return np.sqrt(G_KPC * M / r_safe)


# ---------------------------------------------------------------------
# Perfil Zhao generalizado
# ---------------------------------------------------------------------

def rho_zhao(r: np.ndarray, p: ZhaoParams) -> np.ndarray:
    """Densidad del perfil Zhao generalizado.

    ρ(r) = ρ_s / [(r/r_s)^γ · (1 + (r/r_s)^α)^((β-γ)/α)]

    Args:
        r: Radio en kpc
        p: Parámetros Zhao

    Returns:
        Densidad en M_sun/kpc³
    """
    r = np.asarray(r, dtype=float)
    x = r / max(p.r_s, 1e-10)
    x_safe = np.maximum(x, 1e-10)

    inner = x_safe ** p.gamma
    outer = (1.0 + x_safe ** p.alpha) ** ((p.beta - p.gamma) / p.alpha)

    return p.rho_s / (inner * outer)


def mass_zhao(r: np.ndarray, p: ZhaoParams) -> np.ndarray:
    """Masa encerrada del perfil Zhao (integración numérica).

    M(r) = ∫₀ʳ 4πr'² ρ(r') dr'

    Args:
        r: Radio en kpc
        p: Parámetros Zhao

    Returns:
        Masa encerrada en M_sun
    """
    r_arr = np.atleast_1d(r)
    M = np.zeros_like(r_arr, dtype=float)

    def integrand(r_prime):
        return 4.0 * np.pi * r_prime**2 * rho_zhao(r_prime, p)

    for i, ri in enumerate(r_arr):
        if ri <= 0:
            M[i] = 0.0
        else:
            M[i], _ = quad(integrand, 0, ri, limit=100)

    return M[0] if np.ndim(r) == 0 else M


def V_zhao(r: np.ndarray, p: ZhaoParams) -> np.ndarray:
    """Velocidad circular para perfil Zhao.

    Args:
        r: Radio en kpc
        p: Parámetros Zhao

    Returns:
        Velocidad circular en km/s
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.maximum(r, 1e-10)
    M = mass_zhao(r, p)
    return np.sqrt(G_KPC * M / r_safe)


# ---------------------------------------------------------------------
# Perfil NFW (referencia ΛCDM)
# ---------------------------------------------------------------------

def rho_nfw(r: np.ndarray, p: NFWParams) -> np.ndarray:
    """Densidad del perfil NFW.

    ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

    Args:
        r: Radio en kpc
        p: Parámetros NFW

    Returns:
        Densidad en M_sun/kpc³
    """
    r = np.asarray(r, dtype=float)
    x = r / max(p.r_s, 1e-10)
    x_safe = np.maximum(x, 1e-10)
    return p.rho_s / (x_safe * (1.0 + x_safe)**2)


def mass_nfw(r: np.ndarray, p: NFWParams) -> np.ndarray:
    """Masa encerrada del perfil NFW (forma analítica).

    M(r) = 4π ρ_s r_s³ [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]

    Args:
        r: Radio en kpc
        p: Parámetros NFW

    Returns:
        Masa encerrada en M_sun
    """
    r = np.asarray(r, dtype=float)
    x = r / max(p.r_s, 1e-10)

    term1 = np.log(1.0 + x)
    term2 = x / (1.0 + x)

    return 4.0 * np.pi * p.rho_s * p.r_s**3 * (term1 - term2)


def V_nfw(r: np.ndarray, p: NFWParams) -> np.ndarray:
    """Velocidad circular para perfil NFW.

    Args:
        r: Radio en kpc
        p: Parámetros NFW

    Returns:
        Velocidad circular en km/s
    """
    r = np.asarray(r, dtype=float)
    r_safe = np.maximum(r, 1e-10)
    M = mass_nfw(r, p)
    return np.sqrt(G_KPC * M / r_safe)


# ---------------------------------------------------------------------
# Dependencia entrópica S_loc
# ---------------------------------------------------------------------

def halo_core_params_from_Sloc(
    S_loc: float,
    p: SlocParams
) -> tuple[float, float]:
    """Calcula parámetros de núcleo desde entropía local.

    ρ₀(S_loc) = ρ_★ · (S_★ / S_loc)^α_ρ
    r_core(S_loc) = r_★ · (S_loc / S_★)^α_r

    Interpretación física:
        - Mayor S_loc → mayor conversión → ρ₀ disminuye, r_core aumenta
        - Menor S_loc → menos relajación → perfiles más concentrados

    Args:
        S_loc: Entropía local del halo
        p: Parámetros de referencia

    Returns:
        (rho0, r_core): Parámetros del perfil
    """
    S_loc_safe = max(S_loc, 1e-10)
    S_ratio = p.S_star / S_loc_safe

    rho0 = p.rho_star * (S_ratio ** p.alpha_rho)
    r_core = p.r_star * ((S_loc_safe / p.S_star) ** p.alpha_r)

    return rho0, r_core


def r_core_mass_relation(
    M: float | np.ndarray,
    z: float = 0.0,
    M_star: float = 1e12,
    z_star: float = 0.0,
    r_star: float = 2.0,
    alpha_r: float = 0.3,
    beta_r: float = 0.1
) -> float | np.ndarray:
    """Relación núcleo-masa parametrizada.

    r_core(M, z) = r_★ (M/M_★)^α_r (1+z/1+z_★)^β_r

    Args:
        M: Masa del halo en M_sun
        z: Redshift
        M_star: Masa de referencia (default 10¹² M_sun)
        z_star: Redshift de referencia
        r_star: Radio de núcleo de referencia en kpc
        alpha_r: Exponente de masa
        beta_r: Exponente de redshift

    Returns:
        Radio de núcleo en kpc
    """
    M = np.asarray(M, dtype=float)
    mass_factor = (M / M_star) ** alpha_r
    z_factor = ((1.0 + z) / (1.0 + z_star)) ** beta_r
    return r_star * mass_factor * z_factor


def burkert_from_Sloc(S_loc: float, p: SlocParams) -> BurkertParams:
    """Crea parámetros Burkert desde entropía local.

    Args:
        S_loc: Entropía local del halo
        p: Parámetros de referencia S_loc

    Returns:
        Parámetros Burkert calibrados
    """
    rho0, r0 = halo_core_params_from_Sloc(S_loc, p)
    return BurkertParams(rho0=rho0, r0=r0)


def zhao_cored_from_Sloc(
    S_loc: float,
    p: SlocParams,
    alpha: float = 2.0,
    beta: float = 3.0
) -> ZhaoParams:
    """Crea parámetros Zhao cored desde entropía local.

    Args:
        S_loc: Entropía local del halo
        p: Parámetros de referencia S_loc
        alpha: Parámetro de transición
        beta: Pendiente externa

    Returns:
        Parámetros Zhao con γ=0 (cored)
    """
    rho_s, r_s = halo_core_params_from_Sloc(S_loc, p)
    return ZhaoParams(rho_s=rho_s, r_s=r_s, alpha=alpha, beta=beta, gamma=0.0)
