"""
Likelihoods Cosmológicos
========================

Este módulo implementa el cálculo de likelihoods combinados para el MCMC:
- Likelihood individual por observable (BAO, H(z), SNe)
- Likelihood total combinado
- Log-likelihood para inferencia bayesiana

El likelihood global se define como:
    L_total ∝ exp(-χ²_total / 2)

donde:
    χ²_total = χ²_BAO + χ²_H(z) + χ²_SNe

Referencias:
    - Documento de Simulaciones Conjuntas
    - MCMC Maestro: procedimiento de ajuste bayesiano
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np

from .bao import BAODataset, BAOCalculator
from .hz import HzDataset, chi2_Hz
from .sne import SNDataset, chi2_sne_simple, chi2_sne_marginalized, mu_from_dL
from .distances import DistanceCalculator


@dataclass
class LikelihoodConfig:
    """
    Configuración del likelihood.

    Attributes:
        use_bao: Incluir BAO
        use_Hz: Incluir H(z)
        use_sne: Incluir SNe
        marginalize_M: Marginalizar sobre M absoluta (SNe)
        r_d: Horizonte de sonido para BAO [Mpc]
    """
    use_bao: bool = True
    use_Hz: bool = True
    use_sne: bool = True
    marginalize_M: bool = True
    r_d: float = 147.09  # Mpc


@dataclass
class LikelihoodResult:
    """
    Resultado del cálculo de likelihood.
    """
    log_likelihood: float
    chi2_total: float
    chi2_bao: Optional[float] = None
    chi2_Hz: Optional[float] = None
    chi2_sne: Optional[float] = None
    n_points_total: int = 0

    @property
    def chi2_reduced(self) -> float:
        """χ² reducido."""
        if self.n_points_total > 0:
            return self.chi2_total / self.n_points_total
        return np.nan


class CombinedLikelihood:
    """
    Likelihood combinado para ajuste cosmológico.

    Combina datos de BAO, H(z) y SNe Ia para calcular el
    likelihood total del modelo MCMC.
    """

    def __init__(
        self,
        config: LikelihoodConfig = None,
        bao_data: BAODataset = None,
        Hz_data: HzDataset = None,
        sne_data: SNDataset = None
    ):
        """
        Inicializa el likelihood combinado.

        Args:
            config: Configuración
            bao_data: Dataset BAO
            Hz_data: Dataset H(z)
            sne_data: Dataset SNe
        """
        self.config = config if config is not None else LikelihoodConfig()
        self.bao_data = bao_data
        self.Hz_data = Hz_data
        self.sne_data = sne_data

        # Contar puntos totales
        self.n_bao = bao_data.n_points if bao_data else 0
        self.n_Hz = Hz_data.n_points if Hz_data else 0
        self.n_sne = sne_data.n_sne if sne_data else 0

    def compute_chi2_bao(
        self,
        dist_calc: DistanceCalculator,
        H_func: Callable
    ) -> float:
        """
        Calcula χ² para BAO.

        Args:
            dist_calc: Calculador de distancias
            H_func: Función H(z)

        Returns:
            χ²_BAO
        """
        if not self.config.use_bao or self.bao_data is None:
            return 0.0

        bao_calc = BAOCalculator(dist_calc, H_func, self.config.r_d)
        chi2 = 0.0

        for point in self.bao_data.points:
            z = point.z_eff

            if point.observable == 'D_V/r_d':
                model = bao_calc.D_V_over_rd(z)
            elif point.observable == 'D_M/r_d':
                model = bao_calc.D_M_over_rd(z)
            elif point.observable == 'D_H/r_d':
                model = bao_calc.D_H_over_rd(z)
            else:
                continue

            chi2 += ((model - point.value) / point.error) ** 2

        return chi2

    def compute_chi2_Hz(self, H_func: Callable) -> float:
        """
        Calcula χ² para H(z).

        Args:
            H_func: Función H(z)

        Returns:
            χ²_H(z)
        """
        if not self.config.use_Hz or self.Hz_data is None:
            return 0.0

        z = self.Hz_data.z_values
        H_model = np.array([H_func(zi) for zi in z])
        H_data = self.Hz_data.H_values
        sigma = self.Hz_data.errors

        return chi2_Hz(H_model, H_data, sigma)

    def compute_chi2_sne(self, dist_calc: DistanceCalculator) -> float:
        """
        Calcula χ² para SNe.

        Args:
            dist_calc: Calculador de distancias

        Returns:
            χ²_SNe
        """
        if not self.config.use_sne or self.sne_data is None:
            return 0.0

        z = self.sne_data.z_values
        mu_model = np.array([mu_from_dL(dist_calc.d_L(zi)) for zi in z])
        mu_data = self.sne_data.mu_values

        if self.sne_data.cov_matrix is not None and self.config.marginalize_M:
            return chi2_sne_marginalized(
                mu_model, mu_data, self.sne_data.cov_matrix
            )
        else:
            return chi2_sne_simple(mu_model, mu_data, self.sne_data.errors)

    def __call__(
        self,
        dist_calc: DistanceCalculator,
        H_func: Callable
    ) -> LikelihoodResult:
        """
        Calcula el likelihood total.

        Args:
            dist_calc: Calculador de distancias
            H_func: Función H(z)

        Returns:
            LikelihoodResult con todos los χ² y log-likelihood
        """
        chi2_bao = self.compute_chi2_bao(dist_calc, H_func)
        chi2_Hz = self.compute_chi2_Hz(H_func)
        chi2_sne = self.compute_chi2_sne(dist_calc)

        chi2_total = chi2_bao + chi2_Hz + chi2_sne
        log_likelihood = -0.5 * chi2_total

        n_total = 0
        if self.config.use_bao:
            n_total += self.n_bao
        if self.config.use_Hz:
            n_total += self.n_Hz
        if self.config.use_sne:
            n_total += self.n_sne

        return LikelihoodResult(
            log_likelihood=log_likelihood,
            chi2_total=chi2_total,
            chi2_bao=chi2_bao if self.config.use_bao else None,
            chi2_Hz=chi2_Hz if self.config.use_Hz else None,
            chi2_sne=chi2_sne if self.config.use_sne else None,
            n_points_total=n_total
        )

    def log_likelihood(
        self,
        dist_calc: DistanceCalculator,
        H_func: Callable
    ) -> float:
        """
        Calcula solo el log-likelihood (para MCMC).

        Args:
            dist_calc: Calculador de distancias
            H_func: Función H(z)

        Returns:
            log(L)
        """
        return self(dist_calc, H_func).log_likelihood


def create_log_likelihood_function(
    combined_likelihood: CombinedLikelihood,
    model_builder: Callable[[np.ndarray], Tuple[DistanceCalculator, Callable]]
) -> Callable[[np.ndarray], float]:
    """
    Crea función de log-likelihood para MCMC.

    Esta función es la que se pasa a emcee u otro sampler.

    Args:
        combined_likelihood: Objeto CombinedLikelihood
        model_builder: Función que toma parámetros y retorna
                      (dist_calc, H_func)

    Returns:
        Función log_likelihood(params) -> float
    """
    def log_likelihood_func(params: np.ndarray) -> float:
        try:
            dist_calc, H_func = model_builder(params)
            return combined_likelihood.log_likelihood(dist_calc, H_func)
        except (ValueError, RuntimeError):
            # Si el modelo falla (ej. parámetros no físicos)
            return -np.inf

    return log_likelihood_func


def log_prior_uniform(
    params: np.ndarray,
    bounds: List[Tuple[float, float]]
) -> float:
    """
    Prior uniforme (flat).

    Args:
        params: Vector de parámetros
        bounds: Lista de (min, max) para cada parámetro

    Returns:
        log(prior) = 0 si dentro de bounds, -inf si fuera
    """
    for i, (p, (pmin, pmax)) in enumerate(zip(params, bounds)):
        if p < pmin or p > pmax:
            return -np.inf
    return 0.0


def log_prior_gaussian(
    params: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray
) -> float:
    """
    Prior gaussiano.

    Args:
        params: Vector de parámetros
        means: Medias del prior
        stds: Desviaciones estándar del prior

    Returns:
        log(prior)
    """
    return -0.5 * np.sum(((params - means) / stds) ** 2)


def log_posterior(
    params: np.ndarray,
    log_likelihood_func: Callable,
    log_prior_func: Callable,
) -> float:
    """
    Calcula el log-posterior.

    log(posterior) = log(likelihood) + log(prior)

    Args:
        params: Vector de parámetros
        log_likelihood_func: Función de log-likelihood
        log_prior_func: Función de log-prior

    Returns:
        log(posterior)
    """
    lp = log_prior_func(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_func(params)


# Exportaciones
__all__ = [
    'LikelihoodConfig',
    'LikelihoodResult',
    'CombinedLikelihood',
    'create_log_likelihood_function',
    'log_prior_uniform',
    'log_prior_gaussian',
    'log_posterior',
]
