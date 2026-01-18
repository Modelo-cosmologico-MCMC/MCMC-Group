"""
Post-procesamiento de Cadenas MCMC
==================================

Este módulo implementa herramientas para el análisis de las cadenas MCMC:
- Diagnósticos de convergencia
- Cálculo de estadísticas
- Generación de corner plots
- Exportación de resultados

Referencias:
    - Documento de Simulaciones Conjuntas
    - Gelman & Rubin (1992): convergencia
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import warnings

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False


@dataclass
class ConvergenceDiagnostics:
    """
    Diagnósticos de convergencia.

    Attributes:
        gelman_rubin: Estadístico R-hat de Gelman-Rubin
        effective_samples: Número efectivo de muestras
        autocorr_time: Tiempo de autocorrelación
        converged: Si se considera convergido
    """
    gelman_rubin: Optional[np.ndarray]
    effective_samples: np.ndarray
    autocorr_time: Optional[np.ndarray]
    converged: bool


def gelman_rubin_statistic(chains: np.ndarray) -> np.ndarray:
    """
    Calcula el estadístico R-hat de Gelman-Rubin.

    R-hat < 1.1 indica convergencia.

    Args:
        chains: Array de forma (n_walkers, n_steps, n_params)

    Returns:
        R-hat para cada parámetro
    """
    n_walkers, n_steps, n_params = chains.shape

    # Media de cada cadena
    chain_means = np.mean(chains, axis=1)  # (n_walkers, n_params)

    # Media global
    global_mean = np.mean(chain_means, axis=0)  # (n_params,)

    # Varianza entre cadenas (B)
    B = n_steps / (n_walkers - 1) * np.sum(
        (chain_means - global_mean) ** 2, axis=0
    )

    # Varianza dentro de cada cadena (W)
    chain_vars = np.var(chains, axis=1, ddof=1)  # (n_walkers, n_params)
    W = np.mean(chain_vars, axis=0)  # (n_params,)

    # Estimador de varianza
    var_hat = (n_steps - 1) / n_steps * W + B / n_steps

    # R-hat
    R_hat = np.sqrt(var_hat / W)

    return R_hat


def effective_sample_size(samples: np.ndarray, autocorr_time: np.ndarray = None) -> np.ndarray:
    """
    Calcula el número efectivo de muestras.

    n_eff = n_samples / (2 * tau)

    donde tau es el tiempo de autocorrelación.

    Args:
        samples: Array de muestras (n_samples, n_params)
        autocorr_time: Tiempo de autocorrelación (opcional, se estima si no se da)

    Returns:
        Número efectivo de muestras para cada parámetro
    """
    n_samples = samples.shape[0]

    if autocorr_time is None:
        # Estimación simple
        autocorr_time = estimate_autocorr_time(samples)

    n_eff = n_samples / (2 * autocorr_time)
    return n_eff


def estimate_autocorr_time(samples: np.ndarray, c: float = 5.0) -> np.ndarray:
    """
    Estima el tiempo de autocorrelación.

    Usa el método de ventana automática de Sokal.

    Args:
        samples: Array de muestras (n_samples, n_params)
        c: Factor de corte

    Returns:
        Tiempo de autocorrelación para cada parámetro
    """
    n_samples, n_params = samples.shape
    tau = np.zeros(n_params)

    for i in range(n_params):
        x = samples[:, i]
        x = x - np.mean(x)

        # Autocorrelación
        n = len(x)
        acf = np.correlate(x, x, mode='full')[n-1:]
        acf = acf / acf[0]

        # Suma hasta que tau * c > índice
        tau_sum = 1.0
        for j in range(1, n):
            tau_sum += 2 * acf[j]
            if j >= c * tau_sum:
                break

        tau[i] = max(tau_sum, 1.0)

    return tau


def check_convergence(
    samples: np.ndarray,
    chains: np.ndarray = None,
    min_effective_samples: int = 100,
    max_gelman_rubin: float = 1.1
) -> ConvergenceDiagnostics:
    """
    Verifica la convergencia de las cadenas MCMC.

    Args:
        samples: Muestras aplanadas
        chains: Cadenas sin aplanar (opcional, para R-hat)
        min_effective_samples: Mínimo n_eff requerido
        max_gelman_rubin: Máximo R-hat permitido

    Returns:
        ConvergenceDiagnostics
    """
    # Tiempo de autocorrelación
    try:
        autocorr = estimate_autocorr_time(samples)
    except Exception:
        autocorr = None

    # Tamaño efectivo
    n_eff = effective_sample_size(samples, autocorr)

    # Gelman-Rubin (requiere cadenas separadas)
    if chains is not None and len(chains.shape) == 3:
        try:
            R_hat = gelman_rubin_statistic(chains)
        except Exception:
            R_hat = None
    else:
        R_hat = None

    # Verificar convergencia
    converged = True

    if np.any(n_eff < min_effective_samples):
        converged = False

    if R_hat is not None and np.any(R_hat > max_gelman_rubin):
        converged = False

    return ConvergenceDiagnostics(
        gelman_rubin=R_hat,
        effective_samples=n_eff,
        autocorr_time=autocorr,
        converged=converged
    )


def compute_statistics(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcula estadísticas de las muestras.

    Args:
        samples: Array (n_samples, n_params)

    Returns:
        Diccionario con estadísticas
    """
    return {
        'mean': np.mean(samples, axis=0),
        'median': np.median(samples, axis=0),
        'std': np.std(samples, axis=0),
        'q16': np.percentile(samples, 16, axis=0),
        'q84': np.percentile(samples, 84, axis=0),
        'q2.5': np.percentile(samples, 2.5, axis=0),
        'q97.5': np.percentile(samples, 97.5, axis=0),
    }


def credible_interval(
    samples: np.ndarray,
    level: float = 0.68
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula intervalo de credibilidad.

    Args:
        samples: Array (n_samples, n_params)
        level: Nivel de credibilidad (0.68 = 1σ, 0.95 = 2σ)

    Returns:
        (lower, upper) para cada parámetro
    """
    alpha = 1 - level
    lower = np.percentile(samples, 100 * alpha / 2, axis=0)
    upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def make_corner_plot(
    samples: np.ndarray,
    labels: List[str],
    truths: np.ndarray = None,
    output_file: str = None,
    **kwargs
) -> Any:
    """
    Genera un corner plot.

    Args:
        samples: Muestras (n_samples, n_params)
        labels: Nombres de los parámetros
        truths: Valores verdaderos (opcional)
        output_file: Archivo de salida (opcional)
        **kwargs: Argumentos adicionales para corner.corner

    Returns:
        Figura de matplotlib
    """
    if not HAS_CORNER:
        raise ImportError("corner no está instalado. Instalar con: pip install corner")

    import matplotlib.pyplot as plt

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        **kwargs
    )

    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def make_trace_plot(
    chains: np.ndarray,
    labels: List[str],
    output_file: str = None
) -> Any:
    """
    Genera un trace plot de las cadenas.

    Args:
        chains: Cadenas (n_walkers, n_steps, n_params)
        labels: Nombres de los parámetros
        output_file: Archivo de salida (opcional)

    Returns:
        Figura de matplotlib
    """
    import matplotlib.pyplot as plt

    n_walkers, n_steps, n_params = chains.shape

    fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params), sharex=True)
    if n_params == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for j in range(min(n_walkers, 50)):  # Máximo 50 walkers para visualización
            ax.plot(chains[j, :, i], alpha=0.3, lw=0.5)
        ax.set_ylabel(labels[i])

    axes[-1].set_xlabel('Step')

    plt.tight_layout()

    if output_file is not None:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')

    return fig


def export_results(
    samples: np.ndarray,
    param_names: List[str],
    output_dir: str,
    prefix: str = 'mcmc'
):
    """
    Exporta resultados a archivos.

    Genera:
    - {prefix}_samples.npy: Muestras
    - {prefix}_summary.txt: Resumen estadístico

    Args:
        samples: Muestras
        param_names: Nombres de parámetros
        output_dir: Directorio de salida
        prefix: Prefijo de archivos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar muestras
    np.save(output_dir / f'{prefix}_samples.npy', samples)

    # Resumen estadístico
    stats = compute_statistics(samples)

    with open(output_dir / f'{prefix}_summary.txt', 'w') as f:
        f.write("MCMC Results Summary\n")
        f.write("=" * 60 + "\n\n")

        for i, name in enumerate(param_names):
            f.write(f"{name}:\n")
            f.write(f"  Mean:   {stats['mean'][i]:.6f}\n")
            f.write(f"  Median: {stats['median'][i]:.6f}\n")
            f.write(f"  Std:    {stats['std'][i]:.6f}\n")
            f.write(f"  68% CI: [{stats['q16'][i]:.6f}, {stats['q84'][i]:.6f}]\n")
            f.write(f"  95% CI: [{stats['q2.5'][i]:.6f}, {stats['q97.5'][i]:.6f}]\n")
            f.write("\n")


# Exportaciones
__all__ = [
    'ConvergenceDiagnostics',
    'gelman_rubin_statistic',
    'effective_sample_size',
    'estimate_autocorr_time',
    'check_convergence',
    'compute_statistics',
    'credible_interval',
    'make_corner_plot',
    'make_trace_plot',
    'export_results',
    'HAS_CORNER',
]
