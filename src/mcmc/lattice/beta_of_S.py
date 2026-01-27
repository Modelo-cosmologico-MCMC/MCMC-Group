"""Acoplo β(S) dependiente de la variable entrópica.

En el MCMC, el acoplo gauge se promueve a función de S:

    β(S) = β₀ + β₁ · exp[-b_S · (S - S₃)]

donde:
    - β₀: Plateau de acoplo en S ≈ S₃-S₄
    - β₁, b_S: Controlan el flujo al acercarse a S₃
    - S₃ = 0.999: Sello volumétrico V₃D

Además se introduce un término tensional efectivo que modifica
la dinámica cerca de las transiciones ontológicas.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# Umbrales ontológicos
S1_PLANCK = 0.009
S2_GUT = 0.099
S3_EW = 0.999
S4_BB = 1.001


@dataclass
class BetaParams:
    """Parámetros para β(S).

    Attributes:
        beta0: Valor base del acoplo
        beta1: Amplitud de la variación
        b_S: Inverso del ancho de transición
        S_ref: Punto de referencia (típico S₃)
        use_running: Si True, incluye running logarítmico
        Lambda_QCD: Escala QCD para running (MeV)
    """
    beta0: float = 6.0
    beta1: float = 1.0
    b_S: float = 100.0
    S_ref: float = S3_EW
    use_running: bool = False
    Lambda_QCD: float = 200.0  # MeV


def beta_of_S(S: float | np.ndarray, params: BetaParams) -> float | np.ndarray:
    """Calcula el acoplo β(S) dependiente de S.

    β(S) = β₀ + β₁ · exp[-b_S · (S - S_ref)]

    En regiones S < S_ref, el acoplo aumenta exponencialmente
    reflejando el acoplamiento más fuerte en las fases tempranas.

    Args:
        S: Variable entrópica
        params: Parámetros del modelo

    Returns:
        Acoplo β(S)
    """
    S = np.asarray(S)
    delta_S = S - params.S_ref
    return params.beta0 + params.beta1 * np.exp(-params.b_S * delta_S)


def g_squared_of_S(S: float | np.ndarray, params: BetaParams,
                   N_color: int = 3) -> float | np.ndarray:
    """Calcula g²(S) desde β(S).

    β = 2N/g² → g² = 2N/β

    Args:
        S: Variable entrópica
        params: Parámetros β(S)
        N_color: Número de colores

    Returns:
        Constante de acoplo g²
    """
    beta = beta_of_S(S, params)
    return 2 * N_color / np.maximum(beta, 1e-10)


def alpha_s_of_S(S: float | np.ndarray, params: BetaParams,
                 N_color: int = 3) -> float | np.ndarray:
    """Calcula α_s(S) = g²/(4π).

    Args:
        S: Variable entrópica
        params: Parámetros β(S)
        N_color: Número de colores

    Returns:
        Constante de estructura fina fuerte
    """
    g2 = g_squared_of_S(S, params, N_color)
    return g2 / (4 * np.pi)


def beta_running_1loop(
    mu: float,
    Lambda_QCD: float = 200.0,
    N_f: int = 3,
    N_c: int = 3
) -> float:
    """Calcula β con running a 1 loop.

    β(μ) = 2N_c / g²(μ)
    g²(μ) = g²_0 / [1 + b₀ g²_0 ln(μ/Λ)]

    donde b₀ = (11N_c - 2N_f) / (48π²)

    Args:
        mu: Escala de energía en MeV
        Lambda_QCD: Escala QCD en MeV
        N_f: Número de sabores activos
        N_c: Número de colores

    Returns:
        Valor de β a la escala μ
    """
    if mu <= Lambda_QCD:
        return 1e10  # Régimen no perturbativo

    # Coeficiente de la función beta a 1 loop
    b0 = (11 * N_c - 2 * N_f) / (48 * np.pi**2)

    # Running de g²
    log_ratio = np.log(mu / Lambda_QCD)
    g2 = 1.0 / (b0 * log_ratio)

    return 2 * N_c / g2


def S_of_energy(
    E: float,
    E_Planck: float = 1.22e19,  # GeV
    E_GUT: float = 1e16,        # GeV
    E_EW: float = 246e-3,       # GeV (escala Higgs)
) -> float:
    """Mapea energía a variable entrópica S.

    Aproximación logarítmica entre los umbrales ontológicos.

    Args:
        E: Energía en GeV
        E_Planck: Escala de Planck
        E_GUT: Escala GUT
        E_EW: Escala electrodébil

    Returns:
        Variable entrópica S aproximada
    """
    if E >= E_Planck:
        return S1_PLANCK
    elif E >= E_GUT:
        # Interpolación logarítmica S₁ → S₂
        log_ratio = np.log(E_Planck / E) / np.log(E_Planck / E_GUT)
        return S1_PLANCK + (S2_GUT - S1_PLANCK) * log_ratio
    elif E >= E_EW:
        # Interpolación logarítmica S₂ → S₃
        log_ratio = np.log(E_GUT / E) / np.log(E_GUT / E_EW)
        return S2_GUT + (S3_EW - S2_GUT) * log_ratio
    else:
        # Post-EW: S → S_BB
        return S3_EW + (S4_BB - S3_EW) * (1 - E / E_EW)


def beta_schedule(
    S_values: np.ndarray,
    params: BetaParams
) -> np.ndarray:
    """Genera un schedule de valores β para un barrido en S.

    Args:
        S_values: Array de valores de S
        params: Parámetros β(S)

    Returns:
        Array de valores β correspondientes
    """
    return beta_of_S(S_values, params)


@dataclass
class TensorialCoupling:
    """Acoplo tensional adicional del Campo de Adrián.

    S_tens(S) = α_tens · Σ_x Φ_ten(x; S)²

    Attributes:
        alpha_tens: Coeficiente del término tensional
        S_transition: Punto de transición principal
        width: Ancho de la transición
    """
    alpha_tens: float = 0.1
    S_transition: float = S3_EW
    width: float = 0.01


def tensorial_correction(
    S: float | np.ndarray,
    coupling: TensorialCoupling
) -> float | np.ndarray:
    """Calcula la corrección tensorial a la acción.

    Esta corrección modela el efecto del Campo de Adrián
    en la dinámica gauge cerca de las transiciones.

    Args:
        S: Variable entrópica
        coupling: Parámetros del acoplo tensorial

    Returns:
        Corrección a la acción
    """
    S = np.asarray(S)
    x = (S - coupling.S_transition) / coupling.width

    # Perfil tipo tanh para transición suave
    phi_ten = np.tanh(x)

    return coupling.alpha_tens * phi_ten**2
