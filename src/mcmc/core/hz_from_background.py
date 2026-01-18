from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BackgroundHz:
    """
    Adaptador Bloque I -> callable H(z).
    Requiere solución de background con arrays:
      - z(S): sol["z"]  (decreciente con S creciente, o viceversa según integración)
      - H(S): sol["H"]
    Se construye una interpolación robusta en z.
    """
    z_grid: np.ndarray
    H_grid: np.ndarray

    @classmethod
    def from_solution(cls, sol: dict) -> "BackgroundHz":
        z = np.asarray(sol["z"], float)
        H = np.asarray(sol["H"], float)

        # Queremos z creciente para np.interp
        idx = np.argsort(z)
        z_sorted = z[idx]
        H_sorted = H[idx]

        # Eliminar duplicados numéricos en z (por seguridad)
        z_unique, unique_idx = np.unique(z_sorted, return_index=True)
        H_unique = H_sorted[unique_idx]

        return cls(z_grid=z_unique, H_grid=H_unique)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)

        # clamp al rango del grid (evita extrapolación no controlada)
        zmin = float(self.z_grid[0])
        zmax = float(self.z_grid[-1])
        zc = np.clip(z, zmin, zmax)

        return np.interp(zc, self.z_grid, self.H_grid)
