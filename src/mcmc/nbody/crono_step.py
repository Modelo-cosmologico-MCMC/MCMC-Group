from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CronosStepParams:
    """
    Parametros para el timestep Cronos en N-body.

    El timestep Cronos modifica el paso de tiempo local
    basandose en la densidad y aceleracion local.

    Attributes:
        eta: Factor de precision (tipico 0.01-0.05)
        rho_c: Densidad critica de referencia
        alpha: Factor de friccion/amortiguamiento
    """
    eta: float = 0.02
    rho_c: float = 1.0
    alpha: float = 1.0


def delta_t_cronos(
    acc_norm: np.ndarray,
    a_scale: float,
    rho: np.ndarray,
    p: CronosStepParams
) -> np.ndarray:
    """
    Calcula el timestep Cronos para cada particula.

    El timestep combina:
    1. Criterio de aceleracion estandar (dt ~ eta/sqrt(|a|))
    2. Modificacion por densidad local (suprime dt en regiones densas)

    Args:
        acc_norm: Norma de la aceleracion por particula (N,)
        a_scale: Factor de escala cosmico
        rho: Densidad local por particula (N,)
        p: Parametros de Cronos

    Returns:
        Timestep para cada particula (N,)
    """
    acc_norm = np.asarray(acc_norm, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Criterio de aceleracion base
    base = p.eta / np.sqrt(np.maximum(acc_norm * max(a_scale, 1e-12), 1e-18))

    # Factor de supresion por densidad
    rho_ratio = rho / max(p.rho_c, 1e-12)
    factor = (1.0 + rho_ratio ** 1.5 / max(p.alpha, 1e-12)) ** (-1.0)

    return base * factor


def global_timestep(dt_particles: np.ndarray, safety: float = 0.5) -> float:
    """
    Calcula un timestep global conservador.

    Toma el minimo de los timesteps individuales
    multiplicado por un factor de seguridad.

    Args:
        dt_particles: Timesteps individuales (N,)
        safety: Factor de seguridad (0 < safety <= 1)

    Returns:
        Timestep global
    """
    return safety * np.min(dt_particles)
