from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RhoLatParams:
    """
    Parametros del canal latente rho_lat(S).

    El canal latente representa energia "sellada" que puede
    liberarse en ciertos rangos de S.

    Attributes:
        enabled: Si el canal esta activo
        amp: Amplitud del canal (cuando enabled)
        S0: Centro de activacion en S
        width: Ancho de la transicion (suavizado)
    """
    enabled: bool = False
    amp: float = 0.0
    S0: float = 0.7
    width: float = 0.05


def rho_lat_of_S(S: np.ndarray, p: RhoLatParams) -> np.ndarray:
    """
    Calcula rho_lat(S) parametrico.

    Usa una funcion sigmoide para modelar la activacion
    gradual del canal latente.

    Args:
        S: Array de valores de entropia
        p: Parametros del canal latente

    Returns:
        Array de rho_lat para cada valor de S
    """
    S = np.asarray(S, dtype=float)
    if not p.enabled:
        return np.zeros_like(S)

    w = max(p.width, 1e-12)
    return p.amp * (1.0 / (1.0 + np.exp(-(S - p.S0) / w)))


def drho_lat_dS(S: np.ndarray, p: RhoLatParams) -> np.ndarray:
    """
    Derivada de rho_lat respecto a S.

    Args:
        S: Array de valores de entropia
        p: Parametros del canal latente

    Returns:
        Array de drho_lat/dS
    """
    S = np.asarray(S, dtype=float)
    if not p.enabled:
        return np.zeros_like(S)

    w = max(p.width, 1e-12)
    sig = 1.0 / (1.0 + np.exp(-(S - p.S0) / w))
    return p.amp * sig * (1.0 - sig) / w
