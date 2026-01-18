from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PreGeometricFieldParams:
    dS: float = 1e-3
    k0: float = 8.0
    center: float = 0.005
    width: float = 0.002
    floor: float = 0.5


class PreGeometricField:
    """Φ_pre(Σn) y tasa k_pre(Σn). MVP: perfil gaussiano centrado en ~Σ4–Σ5."""

    def __init__(self, params: PreGeometricFieldParams) -> None:
        self.p = params

    def phi(self, S: float) -> float:
        # curvatura ontológica pre-geométrica (adimensional) - pico en el tramo medio
        x = (S - self.p.center) / max(self.p.width, 1e-12)
        return self.p.k0 * math.exp(-0.5 * x * x)

    def collapse_rate(self, S: float) -> float:
        # tasa k_pre >= floor
        return max(self.p.floor, 1.0 + 0.1 * self.phi(S))
