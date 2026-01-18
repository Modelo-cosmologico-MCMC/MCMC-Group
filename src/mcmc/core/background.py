from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from mcmc.core.cronoshapes import (
    CronosShapeParams,
    C_of_S as C_of_S_shapes,
    T_of_S as T_of_S_shapes,
    N_of_S,
    Phi_ten_of_S,
)


@dataclass
class BackgroundParams:
    """
    Parametros del nucleo Bloque I.

    Integra a(S), t_rel(S) y produce z(S), H(S) usando la Ley de Cronos
    refinada con formas C(S), T(S), Phi_ten(S), N(S).

    Attributes:
        H0: Constante de Hubble [km/s/Mpc]
        shapes: Parametros de las formas de Cronos
    """
    H0: float = 67.4  # km/s/Mpc
    shapes: CronosShapeParams = field(default_factory=CronosShapeParams)


def solve_background(
    S: np.ndarray,
    p: BackgroundParams,
    *,
    S1: float = 0.010,
    S2: float = 0.100,
    S3: float = 1.000,
    S4: float = 1.001,
) -> dict:
    """
    Integra las ecuaciones de fondo desde S4 hacia atras.

    Ecuaciones:
        d(ln a)/dS = C(S)
        d(t_rel)/dS = T(S) * N(S)

    Normalizacion en S4:
        a(S4) = 1
        t_rel(S4) = 0
        H(S4) = H0

    Definiciones derivadas:
        z(S) = 1/a(S) - 1
        H(S) = H0 * C(S)/C(S4)

    Args:
        S: Array de valores de entropia (debe incluir S4 al final)
        p: Parametros del fondo
        S1, S2, S3, S4: Sellos ontologicos

    Returns:
        Diccionario con S, a, z, t_rel, H, C, T, N, Phi_ten
    """
    S = np.asarray(S, dtype=float)

    # Calcular formas de Cronos
    C = C_of_S_shapes(S, p.shapes, S2=S2, S3=S3)
    T = T_of_S_shapes(S, p.shapes, S1=S1, S2=S2, S3=S3)
    N = N_of_S(S, p.shapes, S1=S1, S2=S2, S3=S3)
    Phi = Phi_ten_of_S(S, p.shapes, S1=S1, S2=S2, S3=S3)

    # Arrays para integracion
    a = np.zeros_like(S, dtype=float)
    t = np.zeros_like(S, dtype=float)

    # Normalizacion en S4 (ultimo punto)
    a[-1] = 1.0
    t[-1] = 0.0

    # Integracion hacia atras (S decrece)
    for i in range(len(S) - 2, -1, -1):
        dS = S[i + 1] - S[i]
        # d(ln a)/dS = C(S) => ln a(S_i) = ln a(S_{i+1}) - C*dS
        a[i] = a[i + 1] * np.exp(-C[i + 1] * dS)
        # dt/dS = T*N
        t[i] = t[i + 1] - (T[i + 1] * N[i + 1]) * dS

    # Redshift
    z = 1.0 / a - 1.0

    # H(S) = H0 * C(S)/C(S4) - nucleo computacional
    H = p.H0 * (C / C[-1])

    return {
        "S": S,
        "a": a,
        "z": z,
        "t_rel": t,
        "H": H,
        "C": C,
        "T": T,
        "N": N,
        "Phi_ten": Phi,
    }
