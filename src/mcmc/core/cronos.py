from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CronosParams:
    """
    Parametros de la Ley de Cronos.

    La Ley de Cronos define la relacion entre la variable entropica S
    y el tiempo cosmico t. Controla como "fluye" el tiempo en funcion
    de la entropia.

    Attributes:
        lambda_c: Escala de activacion (controla la transicion)
        k_alpha: Normalizacion/escala de reloj
    """
    lambda_c: float = 0.05
    k_alpha: float = 1.0


def dt_dS(S: np.ndarray, p: CronosParams) -> np.ndarray:
    """
    Calcula dt/dS segun la Ley de Cronos.

    Forma generica: dt/dS ~ tanh(S/lambda_c).
    Esta forma asegura que el tiempo "arranca" suavemente
    desde S=0 y satura a valores grandes de S.

    Args:
        S: Array de valores de entropia
        p: Parametros de Cronos

    Returns:
        Array de dt/dS para cada valor de S
    """
    x = np.asarray(S, dtype=float) / max(p.lambda_c, 1e-12)
    return p.k_alpha * np.tanh(x)


def t_of_S(S: np.ndarray, p: CronosParams) -> np.ndarray:
    """
    Integra la Ley de Cronos para obtener t(S).

    Args:
        S: Array de valores de entropia (debe ser estrictamente creciente)
        p: Parametros de Cronos

    Returns:
        Array de tiempo cosmico t para cada valor de S

    Raises:
        ValueError: Si S no es estrictamente creciente
    """
    S = np.asarray(S, dtype=float)
    dS = np.diff(S)
    if np.any(dS <= 0):
        raise ValueError("S debe ser estrictamente creciente")

    mid = 0.5 * (S[1:] + S[:-1])
    integrand = dt_dS(mid, p)
    t = np.zeros_like(S)
    t[1:] = np.cumsum(integrand * dS)
    return t


def S_of_t(t: np.ndarray, S_grid: np.ndarray, p: CronosParams) -> np.ndarray:
    """
    Invierte la Ley de Cronos para obtener S(t).

    Args:
        t: Array de tiempos cosmicos
        S_grid: Grid de S para interpolar
        p: Parametros de Cronos

    Returns:
        Array de S para cada valor de t
    """
    from scipy.interpolate import interp1d

    t_grid = t_of_S(S_grid, p)
    interp = interp1d(t_grid, S_grid, kind="cubic", bounds_error=False, fill_value="extrapolate")
    return interp(t)
