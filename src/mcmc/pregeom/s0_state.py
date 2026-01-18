from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class PreGeomParams:
    """
    Parametros del estado pre-geometrico (Bloque 0).

    Attributes:
        eps: Imperfeccion primordial (Ep0 ~ eps)
        phi0: Condicion inicial campo tensional
        k0: Rigidez pre-geometrica efectiva
        S_start: Punto de entrada al Bloque I
    """
    eps: float = 1e-2
    phi0: float = 0.0
    k0: float = 1.0
    S_start: float = 0.010


@dataclass(frozen=True)
class InitialConditions:
    """
    Condiciones iniciales para el Bloque I.

    Estas condiciones se derivan del estado pre-geometrico
    y sirven como contrato entre Bloque 0 y Bloque I.
    """
    Mp_pre: float
    Ep_pre: float
    phi_pre: float
    k_pre: float
    S_start: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_initial_conditions(p: PreGeomParams) -> InitialConditions:
    """
    Calcula las condiciones iniciales para el Bloque I.

    El modelo minimo reparte Mp/Ep pre en torno a eps.
    Ajusta aqui si tienes ley Mp(S), Ep(S) en S in [0.001, 0.009].

    Args:
        p: Parametros pre-geometricos

    Returns:
        InitialConditions para Bloque I

    Raises:
        ValueError: Si eps no esta en (0, 0.5)
    """
    if not (0.0 < p.eps < 0.5):
        raise ValueError("eps debe estar en (0, 0.5)")

    Ep_pre = float(p.eps)
    Mp_pre = float(1.0 - Ep_pre)

    return InitialConditions(
        Mp_pre=Mp_pre,
        Ep_pre=Ep_pre,
        phi_pre=float(p.phi0),
        k_pre=float(p.k0),
        S_start=float(p.S_start),
    )
