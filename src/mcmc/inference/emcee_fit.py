"""
Ajuste Bayesiano con emcee
==========================

Este módulo implementa el ajuste bayesiano del MCMC usando el sampler
affine-invariant emcee.

El flujo de trabajo es:
1. Definir parámetros y priors
2. Construir función de log-posterior
3. Ejecutar cadenas MCMC
4. Analizar convergencia y extraer posteriors

Referencias:
    - Documento de Simulaciones Conjuntas
    - MCMC Maestro: procedimiento de ajuste
    - Foreman-Mackey et al. (2013), emcee paper
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
import numpy as np
from pathlib import Path
import warnings

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee no está instalado. Instalar con: pip install emcee")


@dataclass
class Parameter:
    """
    Definición de un parámetro del modelo.

    Attributes:
        name: Nombre del parámetro
        latex: Nombre en LaTeX para plots
        initial: Valor inicial
        prior_min: Límite inferior del prior uniforme
        prior_max: Límite superior del prior uniforme
        fixed: Si True, el parámetro está fijo (no se muestrea)
        gaussian_prior: Si no es None, (mean, std) para prior gaussiano
    """
    name: str
    latex: str
    initial: float
    prior_min: float
    prior_max: float
    fixed: bool = False
    gaussian_prior: Optional[Tuple[float, float]] = None


@dataclass
class MCMCConfig:
    """
    Configuración del MCMC.

    Attributes:
        n_walkers: Número de walkers
        n_steps: Número de pasos por walker
        n_burnin: Número de pasos de burn-in a descartar
        thin: Factor de thinning
        moves: Movimiento de emcee (None = default)
        progress: Mostrar barra de progreso
        backend_file: Archivo para guardar cadenas (None = no guardar)
    """
    n_walkers: int = 32
    n_steps: int = 5000
    n_burnin: int = 1000
    thin: int = 1
    moves: Any = None
    progress: bool = True
    backend_file: Optional[str] = None


@dataclass
class MCMCResult:
    """
    Resultado del ajuste MCMC.

    Attributes:
        samples: Array de muestras (n_samples, n_params)
        param_names: Nombres de los parámetros
        log_prob: Log-probabilidades de cada muestra
        acceptance_fraction: Fracción de aceptación
        autocorr_time: Tiempo de autocorrelación estimado
        best_fit: Parámetros del mejor ajuste
        mean: Medias de los parámetros
        std: Desviaciones estándar
        quantiles: Percentiles (16, 50, 84)
    """
    samples: np.ndarray
    param_names: List[str]
    log_prob: np.ndarray
    acceptance_fraction: float
    autocorr_time: Optional[np.ndarray]
    best_fit: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    quantiles: Dict[str, np.ndarray]

    def get_param_summary(self, name: str) -> Dict[str, float]:
        """Obtiene resumen de un parámetro."""
        idx = self.param_names.index(name)
        return {
            'best': self.best_fit[idx],
            'mean': self.mean[idx],
            'std': self.std[idx],
            'median': self.quantiles['50'][idx],
            'lower': self.quantiles['16'][idx],
            'upper': self.quantiles['84'][idx],
        }

    def summary(self) -> str:
        """Genera resumen textual de los resultados."""
        lines = ["=" * 60]
        lines.append("RESULTADOS DEL AJUSTE MCMC")
        lines.append("=" * 60)

        for i, name in enumerate(self.param_names):
            q16, q50, q84 = self.quantiles['16'][i], self.quantiles['50'][i], self.quantiles['84'][i]
            err_up = q84 - q50
            err_down = q50 - q16
            lines.append(f"{name}: {q50:.4f} +{err_up:.4f} -{err_down:.4f}")

        lines.append("-" * 60)
        lines.append(f"Best-fit log(L): {np.max(self.log_prob):.2f}")
        lines.append(f"Acceptance fraction: {self.acceptance_fraction:.2%}")
        if self.autocorr_time is not None:
            lines.append(f"Autocorr time (mean): {np.mean(self.autocorr_time):.1f}")

        return "\n".join(lines)


class MCMCFitter:
    """
    Clase principal para ajuste MCMC con emcee.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        log_likelihood: Callable[[np.ndarray], float],
        config: MCMCConfig = None
    ):
        """
        Inicializa el fitter.

        Args:
            parameters: Lista de parámetros
            log_likelihood: Función de log-likelihood
            config: Configuración del MCMC
        """
        if not HAS_EMCEE:
            raise ImportError("emcee es requerido. Instalar con: pip install emcee")

        self.parameters = [p for p in parameters if not p.fixed]
        self.all_parameters = parameters
        self.log_likelihood = log_likelihood
        self.config = config if config is not None else MCMCConfig()

        self.n_dim = len(self.parameters)
        self.param_names = [p.name for p in self.parameters]

        # Construir función de prior
        self._build_prior()

    def _build_prior(self):
        """Construye la función de log-prior."""
        self.bounds = [(p.prior_min, p.prior_max) for p in self.parameters]
        self.gaussian_priors = [p.gaussian_prior for p in self.parameters]

    def log_prior(self, params: np.ndarray) -> float:
        """Calcula el log-prior."""
        for i, (p, (pmin, pmax)) in enumerate(zip(params, self.bounds)):
            if p < pmin or p > pmax:
                return -np.inf

        # Priors gaussianos adicionales
        log_p = 0.0
        for i, gp in enumerate(self.gaussian_priors):
            if gp is not None:
                mean, std = gp
                log_p -= 0.5 * ((params[i] - mean) / std) ** 2

        return log_p

    def log_posterior(self, params: np.ndarray) -> float:
        """Calcula el log-posterior."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def _initial_positions(self) -> np.ndarray:
        """Genera posiciones iniciales para los walkers."""
        initial = np.array([p.initial for p in self.parameters])

        # Dispersión inicial: 1% del rango del prior
        scales = np.array([
            0.01 * (p.prior_max - p.prior_min)
            for p in self.parameters
        ])

        pos = initial + scales * np.random.randn(self.config.n_walkers, self.n_dim)

        # Asegurar que están dentro de los bounds
        for i, (pmin, pmax) in enumerate(self.bounds):
            pos[:, i] = np.clip(pos[:, i], pmin, pmax)

        return pos

    def run(self, initial_pos: np.ndarray = None) -> MCMCResult:
        """
        Ejecuta el MCMC.

        Args:
            initial_pos: Posiciones iniciales (opcional)

        Returns:
            MCMCResult con las muestras y estadísticas
        """
        # Posiciones iniciales
        if initial_pos is None:
            pos = self._initial_positions()
        else:
            pos = initial_pos

        # Backend para guardar cadenas
        backend = None
        if self.config.backend_file is not None:
            backend = emcee.backends.HDFBackend(self.config.backend_file)
            backend.reset(self.config.n_walkers, self.n_dim)

        # Crear sampler
        sampler = emcee.EnsembleSampler(
            self.config.n_walkers,
            self.n_dim,
            self.log_posterior,
            moves=self.config.moves,
            backend=backend
        )

        # Ejecutar
        sampler.run_mcmc(
            pos,
            self.config.n_steps,
            progress=self.config.progress
        )

        # Procesar resultados
        return self._process_results(sampler)

    def _process_results(self, sampler) -> MCMCResult:
        """Procesa los resultados del sampler."""
        # Descartar burn-in
        flat_samples = sampler.get_chain(
            discard=self.config.n_burnin,
            thin=self.config.thin,
            flat=True
        )

        log_prob = sampler.get_log_prob(
            discard=self.config.n_burnin,
            thin=self.config.thin,
            flat=True
        )

        # Mejor ajuste
        best_idx = np.argmax(log_prob)
        best_fit = flat_samples[best_idx]

        # Estadísticas
        mean = np.mean(flat_samples, axis=0)
        std = np.std(flat_samples, axis=0)

        q16, q50, q84 = np.percentile(flat_samples, [16, 50, 84], axis=0)

        # Tiempo de autocorrelación
        try:
            autocorr = sampler.get_autocorr_time(quiet=True)
        except Exception:
            autocorr = None

        return MCMCResult(
            samples=flat_samples,
            param_names=self.param_names,
            log_prob=log_prob,
            acceptance_fraction=np.mean(sampler.acceptance_fraction),
            autocorr_time=autocorr,
            best_fit=best_fit,
            mean=mean,
            std=std,
            quantiles={'16': q16, '50': q50, '84': q84}
        )


def run_mcmc_fit(
    parameters: List[Parameter],
    log_likelihood: Callable[[np.ndarray], float],
    n_walkers: int = 32,
    n_steps: int = 5000,
    n_burnin: int = 1000,
    progress: bool = True
) -> MCMCResult:
    """
    Función de conveniencia para ejecutar MCMC.

    Args:
        parameters: Lista de parámetros
        log_likelihood: Función de log-likelihood
        n_walkers: Número de walkers
        n_steps: Número de pasos
        n_burnin: Pasos de burn-in
        progress: Mostrar progreso

    Returns:
        MCMCResult
    """
    config = MCMCConfig(
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burnin=n_burnin,
        progress=progress
    )

    fitter = MCMCFitter(parameters, log_likelihood, config)
    return fitter.run()


# Parámetros predefinidos para el MCMC refinado
def get_mcmc_refined_parameters() -> List[Parameter]:
    """
    Retorna parámetros del modelo MCMC refinado (Nivel A).

    Parámetros: H0, Omega_id0, z_trans, epsilon, gamma
    """
    return [
        Parameter(
            name='H0',
            latex=r'$H_0$',
            initial=67.4,
            prior_min=60.0,
            prior_max=80.0
        ),
        Parameter(
            name='Omega_id0',
            latex=r'$\Omega_{id,0}$',
            initial=0.7,
            prior_min=0.5,
            prior_max=0.9
        ),
        Parameter(
            name='z_trans',
            latex=r'$z_{trans}$',
            initial=0.5,
            prior_min=0.1,
            prior_max=2.0
        ),
        Parameter(
            name='epsilon',
            latex=r'$\epsilon$',
            initial=0.01,
            prior_min=-0.1,
            prior_max=0.2
        ),
        Parameter(
            name='gamma',
            latex=r'$\gamma$',
            initial=0.0,
            prior_min=-1.0,
            prior_max=1.0
        ),
    ]


# Exportaciones
__all__ = [
    'Parameter',
    'MCMCConfig',
    'MCMCResult',
    'MCMCFitter',
    'run_mcmc_fit',
    'get_mcmc_refined_parameters',
    'HAS_EMCEE',
]
