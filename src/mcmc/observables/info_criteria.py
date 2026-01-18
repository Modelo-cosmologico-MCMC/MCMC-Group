"""
Criterios de Información
========================

Este módulo implementa los criterios de información para comparación
de modelos:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- DIC (Deviance Information Criterion)

Estos criterios permiten comparar el MCMC con ΛCDM y otros modelos,
penalizando la complejidad adicional.

Referencias:
    - Documento de Simulaciones Conjuntas: evaluación AIC/BIC
    - Akaike (1974), Schwarz (1978)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class InformationCriteria:
    """
    Resultados de criterios de información.

    Attributes:
        chi2: χ² mínimo (best-fit)
        n_data: Número de puntos de datos
        n_params: Número de parámetros libres
        AIC: Akaike Information Criterion
        BIC: Bayesian Information Criterion
        AICc: AIC corregido para muestras pequeñas
    """
    chi2: float
    n_data: int
    n_params: int
    AIC: float
    BIC: float
    AICc: float

    def __str__(self) -> str:
        return (
            f"Information Criteria:\n"
            f"  χ² = {self.chi2:.2f}\n"
            f"  n_data = {self.n_data}, n_params = {self.n_params}\n"
            f"  AIC  = {self.AIC:.2f}\n"
            f"  AICc = {self.AICc:.2f}\n"
            f"  BIC  = {self.BIC:.2f}"
        )


def compute_AIC(chi2: float, n_params: int) -> float:
    """
    Calcula el Akaike Information Criterion.

    AIC = χ² + 2k

    donde k es el número de parámetros libres.

    Un AIC menor indica un mejor modelo (balance entre ajuste y complejidad).

    Args:
        chi2: χ² mínimo del ajuste
        n_params: Número de parámetros libres

    Returns:
        AIC
    """
    return chi2 + 2 * n_params


def compute_AICc(chi2: float, n_params: int, n_data: int) -> float:
    """
    Calcula el AIC corregido para muestras pequeñas.

    AICc = AIC + 2k(k+1) / (n - k - 1)

    Esta corrección es importante cuando n_data / n_params < 40.

    Args:
        chi2: χ² mínimo
        n_params: Número de parámetros
        n_data: Número de datos

    Returns:
        AICc
    """
    AIC = compute_AIC(chi2, n_params)

    if n_data - n_params - 1 > 0:
        correction = 2 * n_params * (n_params + 1) / (n_data - n_params - 1)
        return AIC + correction
    else:
        return np.inf


def compute_BIC(chi2: float, n_params: int, n_data: int) -> float:
    """
    Calcula el Bayesian Information Criterion.

    BIC = χ² + k ln(n)

    El BIC penaliza más fuertemente los parámetros adicionales
    que el AIC para muestras grandes.

    Args:
        chi2: χ² mínimo
        n_params: Número de parámetros
        n_data: Número de datos

    Returns:
        BIC
    """
    return chi2 + n_params * np.log(n_data)


def compute_all_criteria(
    chi2: float,
    n_params: int,
    n_data: int
) -> InformationCriteria:
    """
    Calcula todos los criterios de información.

    Args:
        chi2: χ² mínimo
        n_params: Número de parámetros
        n_data: Número de datos

    Returns:
        InformationCriteria con AIC, BIC, AICc
    """
    return InformationCriteria(
        chi2=chi2,
        n_data=n_data,
        n_params=n_params,
        AIC=compute_AIC(chi2, n_params),
        BIC=compute_BIC(chi2, n_params, n_data),
        AICc=compute_AICc(chi2, n_params, n_data)
    )


def delta_AIC(AIC_model: float, AIC_reference: float) -> float:
    """
    Calcula ΔAIC respecto a un modelo de referencia.

    ΔAIC = AIC_model - AIC_reference

    Interpretación:
    - ΔAIC < 2: Modelos igualmente plausibles
    - 2 < ΔAIC < 7: Evidencia moderada contra el modelo
    - ΔAIC > 10: Evidencia fuerte contra el modelo

    Args:
        AIC_model: AIC del modelo a evaluar
        AIC_reference: AIC del modelo de referencia (ej. ΛCDM)

    Returns:
        ΔAIC
    """
    return AIC_model - AIC_reference


def delta_BIC(BIC_model: float, BIC_reference: float) -> float:
    """
    Calcula ΔBIC respecto a un modelo de referencia.

    ΔBIC = BIC_model - BIC_reference

    Interpretación (escala de Jeffreys):
    - ΔBIC < 2: Evidencia insignificante
    - 2 < ΔBIC < 6: Evidencia positiva contra el modelo
    - 6 < ΔBIC < 10: Evidencia fuerte contra el modelo
    - ΔBIC > 10: Evidencia muy fuerte contra el modelo

    Args:
        BIC_model: BIC del modelo a evaluar
        BIC_reference: BIC del modelo de referencia

    Returns:
        ΔBIC
    """
    return BIC_model - BIC_reference


def bayes_factor_from_BIC(delta_BIC: float) -> float:
    """
    Estima el factor de Bayes desde ΔBIC.

    ln(B) ≈ -ΔBIC / 2

    donde B = P(data|model) / P(data|reference)

    Args:
        delta_BIC: ΔBIC entre modelos

    Returns:
        Factor de Bayes aproximado
    """
    return np.exp(-delta_BIC / 2)


def akaike_weights(AIC_values: np.ndarray) -> np.ndarray:
    """
    Calcula los pesos de Akaike para un conjunto de modelos.

    w_i = exp(-ΔAIC_i / 2) / Σ exp(-ΔAIC_j / 2)

    Los pesos suman 1 y representan la probabilidad relativa
    de cada modelo.

    Args:
        AIC_values: Array con AIC de cada modelo

    Returns:
        Array con pesos de Akaike
    """
    delta_AIC = AIC_values - np.min(AIC_values)
    exp_terms = np.exp(-delta_AIC / 2)
    return exp_terms / np.sum(exp_terms)


@dataclass
class ModelComparison:
    """
    Comparación entre dos modelos.
    """
    model_name: str
    reference_name: str
    delta_AIC: float
    delta_BIC: float
    bayes_factor: float
    interpretation: str


def compare_models(
    model_criteria: InformationCriteria,
    reference_criteria: InformationCriteria,
    model_name: str = "MCMC",
    reference_name: str = "ΛCDM"
) -> ModelComparison:
    """
    Compara dos modelos usando criterios de información.

    Args:
        model_criteria: Criterios del modelo a evaluar
        reference_criteria: Criterios del modelo de referencia
        model_name: Nombre del modelo
        reference_name: Nombre de la referencia

    Returns:
        ModelComparison con la comparación completa
    """
    d_AIC = delta_AIC(model_criteria.AIC, reference_criteria.AIC)
    d_BIC = delta_BIC(model_criteria.BIC, reference_criteria.BIC)
    bf = bayes_factor_from_BIC(d_BIC)

    # Interpretación
    if d_BIC < -10:
        interp = f"Evidencia muy fuerte a favor de {model_name}"
    elif d_BIC < -6:
        interp = f"Evidencia fuerte a favor de {model_name}"
    elif d_BIC < -2:
        interp = f"Evidencia positiva a favor de {model_name}"
    elif d_BIC < 2:
        interp = "Modelos igualmente plausibles"
    elif d_BIC < 6:
        interp = f"Evidencia positiva a favor de {reference_name}"
    elif d_BIC < 10:
        interp = f"Evidencia fuerte a favor de {reference_name}"
    else:
        interp = f"Evidencia muy fuerte a favor de {reference_name}"

    return ModelComparison(
        model_name=model_name,
        reference_name=reference_name,
        delta_AIC=d_AIC,
        delta_BIC=d_BIC,
        bayes_factor=bf,
        interpretation=interp
    )


# ΛCDM tiene 2 parámetros relevantes para late-time cosmology: H0, Ω_m
N_PARAMS_LCDM = 2


def get_lcdm_criteria(chi2: float, n_data: int) -> InformationCriteria:
    """
    Calcula criterios para ΛCDM como referencia.

    Args:
        chi2: χ² del mejor ajuste ΛCDM
        n_data: Número de datos

    Returns:
        InformationCriteria para ΛCDM
    """
    return compute_all_criteria(chi2, N_PARAMS_LCDM, n_data)


# Exportaciones
__all__ = [
    'InformationCriteria',
    'compute_AIC',
    'compute_AICc',
    'compute_BIC',
    'compute_all_criteria',
    'delta_AIC',
    'delta_BIC',
    'bayes_factor_from_BIC',
    'akaike_weights',
    'ModelComparison',
    'compare_models',
    'N_PARAMS_LCDM',
    'get_lcdm_criteria',
]
