from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PoissonParams:
    """
    Parametros para el solver de Poisson.

    Attributes:
        G: Constante gravitacional (unidades del sistema)
        softening: Suavizado para evitar singularidades
        box_size: Tamano de la caja (para condiciones periodicas)
    """
    G: float = 1.0
    softening: float = 0.01
    box_size: float | None = None


def direct_acceleration(x: np.ndarray, m: np.ndarray | None, p: PoissonParams) -> np.ndarray:
    """
    Calcula aceleraciones por suma directa (O(N^2)).

    Util para sistemas pequenos o como referencia.

    Args:
        x: Posiciones (N, 3)
        m: Masas (N,) o None (masas unitarias)
        p: Parametros de Poisson

    Returns:
        Aceleraciones (N, 3)
    """
    N = x.shape[0]
    if m is None:
        m = np.ones(N)

    acc = np.zeros_like(x)
    eps2 = p.softening ** 2

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx = x[j] - x[i]

            # Condiciones periodicas si aplica
            if p.box_size is not None:
                dx = dx - p.box_size * np.round(dx / p.box_size)

            r2 = np.sum(dx ** 2) + eps2
            r = np.sqrt(r2)
            acc[i] += p.G * m[j] * dx / (r * r2)

    return acc


def estimate_density(x: np.ndarray, m: np.ndarray | None, h: float) -> np.ndarray:
    """
    Estima la densidad local usando un kernel SPH simplificado.

    Args:
        x: Posiciones (N, 3)
        m: Masas (N,) o None
        h: Radio del kernel

    Returns:
        Densidad estimada por particula (N,)
    """
    N = x.shape[0]
    if m is None:
        m = np.ones(N)

    rho = np.zeros(N)
    h2 = h ** 2

    for i in range(N):
        for j in range(N):
            dx = x[j] - x[i]
            r2 = np.sum(dx ** 2)
            if r2 < h2:
                # Kernel cubico simplificado
                q = np.sqrt(r2) / h
                W = (1.0 - q) ** 3 if q < 1.0 else 0.0
                rho[i] += m[j] * W

    # Normalizacion aproximada
    norm = 1.0 / (np.pi * h ** 3)
    return rho * norm


def make_acceleration_fn(m: np.ndarray | None, p: PoissonParams):
    """
    Crea una funcion de aceleracion para usar con el integrador.

    Args:
        m: Masas de las particulas
        p: Parametros de Poisson

    Returns:
        Funcion acc(x) -> aceleraciones
    """
    def acc_fn(x: np.ndarray) -> np.ndarray:
        return direct_acceleration(x, m, p)

    return acc_fn
