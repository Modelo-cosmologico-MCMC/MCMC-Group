from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RhoIDParams:
    """
    Parametrizacion refinada (MVP):
      rho_id(z) ~ rho0*(1+z)^3 para z>z_trans
      rho_id(z) ~ rho0*[1 + eps*(z_trans - z)] para z<=z_trans
    """
    rho0: float = 0.7
    z_trans: float = 0.6
    eps: float = 0.05


def rho_id_of_z(z: np.ndarray, p: RhoIDParams) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    high = z > p.z_trans
    out[high] = p.rho0 * (1.0 + z[high]) ** 3
    out[~high] = p.rho0 * (1.0 + p.eps * (p.z_trans - z[~high]))
    return out
