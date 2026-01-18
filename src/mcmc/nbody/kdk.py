from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class State:
    """
    Estado de un sistema de N particulas.

    Attributes:
        x: Posiciones (N, 3)
        v: Velocidades (N, 3)
        m: Masas (N,) - opcional, default todas iguales
    """
    x: np.ndarray
    v: np.ndarray
    m: np.ndarray | None = None

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if self.m is not None:
            self.m = np.asarray(self.m, dtype=float)

    @property
    def N(self) -> int:
        return self.x.shape[0]

    def kinetic_energy(self) -> float:
        """Calcula la energia cinetica total."""
        v2 = np.sum(self.v ** 2, axis=1)
        if self.m is None:
            return 0.5 * np.sum(v2)
        return 0.5 * np.sum(self.m * v2)


AccelerationFn = Callable[[np.ndarray], np.ndarray]


def kdk_step(state: State, dt: float, acc_fn: AccelerationFn) -> State:
    """
    Integrador leapfrog Kick-Drift-Kick (KDK).

    Este es el integrador simplectico estandar para N-body.
    Preserva la estructura del espacio de fases.

    Args:
        state: Estado actual del sistema
        dt: Paso de tiempo
        acc_fn: Funcion que calcula aceleraciones dado x

    Returns:
        Nuevo estado despues de un paso dt
    """
    # Kick (medio paso)
    a0 = acc_fn(state.x)
    v_half = state.v + 0.5 * dt * a0

    # Drift (paso completo)
    x1 = state.x + dt * v_half

    # Kick (medio paso)
    a1 = acc_fn(x1)
    v1 = v_half + 0.5 * dt * a1

    return State(x=x1, v=v1, m=state.m)


def integrate(
    state: State,
    acc_fn: AccelerationFn,
    dt: float,
    n_steps: int,
    save_every: int = 1
) -> list[State]:
    """
    Integra el sistema por n_steps pasos.

    Args:
        state: Estado inicial
        acc_fn: Funcion de aceleracion
        dt: Paso de tiempo
        n_steps: Numero de pasos
        save_every: Guardar estado cada N pasos

    Returns:
        Lista de estados guardados
    """
    states = [state]
    current = state

    for i in range(n_steps):
        current = kdk_step(current, dt, acc_fn)
        if (i + 1) % save_every == 0:
            states.append(current)

    return states
