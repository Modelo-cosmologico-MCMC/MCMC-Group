from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RhoIDRefinedParams:
    """
    rho_id(z) refinada con transicion (modelo efectivo de validacion).

    Parametrizacion por tramos:
      - z > z_trans:  rho0 * (1+z)^3
      - z <= z_trans: rho0 * [1 + eps*(z_trans - z)]

    Nota ontologica: el regimen tardio se aproxima a un "vacio" casi constante,
    regulado por la transicion; el regimen alto-z preserva el caracter de
    contribucion de tipo materia efectiva en el cierre.

    Attributes:
        rho0: Amplitud normalizada del canal indeterminado
        z_trans: Redshift de transicion
        eps: Pendiente en el regimen tardio (z <= z_trans)
    """
    rho0: float = 0.70
    z_trans: float = 1.00
    eps: float = 0.05


def rho_id_refined(z: np.ndarray, p: RhoIDRefinedParams) -> np.ndarray:
    """
    Calcula rho_id(z) con transicion suave.

    Para z > z_trans: rho_id ~ (1+z)^3 (comportamiento tipo materia)
    Para z <= z_trans: rho_id ~ constante + pequena pendiente (tipo vacio)

    Args:
        z: Array de redshifts
        p: Parametros de rho_id refinada

    Returns:
        Array de rho_id(z)
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    hi = z > p.z_trans
    out[hi] = p.rho0 * (1.0 + z[hi]) ** 3

    out[~hi] = p.rho0 * (1.0 + p.eps * (p.z_trans - z[~hi]))

    # seguridad numerica: nunca negativo en el cierre efectivo
    return np.maximum(out, 1e-18)


def drho_id_dz(z: np.ndarray, p: RhoIDRefinedParams) -> np.ndarray:
    """
    Derivada de rho_id respecto a z.

    Args:
        z: Array de redshifts
        p: Parametros de rho_id refinada

    Returns:
        Array de drho_id/dz
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    hi = z > p.z_trans
    out[hi] = 3.0 * p.rho0 * (1.0 + z[hi]) ** 2

    out[~hi] = -p.rho0 * p.eps

    return out
