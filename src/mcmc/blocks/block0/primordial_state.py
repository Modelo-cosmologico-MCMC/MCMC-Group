from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimordialState:
    """Estado Primordial S0: unidad dual irreductible Mp/Ep sin geometría interna."""
    Mp0: float  # ~ 1 - eps
    Ep0: float  # ~ eps
    eps: float  # eps pequeño > 0

    def __post_init__(self) -> None:
        if not (0.0 < self.eps < 0.5):
            raise ValueError("eps debe estar en (0, 0.5).")
        if abs((self.Mp0 + self.Ep0) - 1.0) > 1e-9:
            raise ValueError("En S0 debe cumplirse Mp0 + Ep0 = 1.")
        if min(self.Mp0, self.Ep0) < 0.0:
            raise ValueError("Mp0 y Ep0 deben ser no negativas.")
