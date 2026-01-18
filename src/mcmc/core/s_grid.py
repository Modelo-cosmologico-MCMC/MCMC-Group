from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Seals:
    """Sellos ontologicos (valores criticos en S)."""
    S1: float = 0.010
    S2: float = 0.100
    S3: float = 1.000
    S4: float = 1.001


@dataclass
class SGrid:
    """Rejilla discreta en entropia S."""
    S_min: float = 0.010
    S_max: float = 1.001
    dS: float = 1e-3
    seals: Seals = Seals()

    def build(self) -> np.ndarray:
        S = np.arange(self.S_min, self.S_max + 0.5 * self.dS, self.dS)
        return S

    def assert_seals_on_grid(self, S: np.ndarray) -> None:
        for name, val in vars(self.seals).items():
            if not np.any(np.isclose(S, val, atol=1e-12)):
                raise ValueError(f"Sello {name}={val} no esta en la rejilla. Ajusta dS o rangos.")


def create_default_grid() -> tuple[SGrid, np.ndarray]:
    grid = SGrid()
    S = grid.build()
    grid.assert_seals_on_grid(S)
    return grid, S
