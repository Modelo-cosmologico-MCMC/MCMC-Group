from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BackgroundParams:
    """
    Parametros minimos del fondo.
    Esta version es un MVP estable; puedes reemplazar C(S), T(S), Phi_ten(S)
    por tu receta completa sin romper la API.
    """
    H0: float = 67.4  # km/s/Mpc

    # Perfil suave: transicion en torno a S3 (ejemplo)
    C0: float = 1.0
    C1: float = 1.0
    S_c: float = 1.0
    width: float = 0.03

    # Cronificacion (tiempo relativo)
    T0: float = 1.0

    # Tensional: controla N(S)=exp(Phi_ten)
    phi_amp: float = 0.0  # empieza neutro para estabilidad


def _smoothstep(x: np.ndarray) -> np.ndarray:
    # sigmoide estable
    return 0.5 * (1.0 + np.tanh(x))


def C_of_S(S: np.ndarray, p: BackgroundParams) -> np.ndarray:
    # C(S) ~ constante (MVP), con ligera transicion si se desea.
    x = (S - p.S_c) / max(p.width, 1e-9)
    return p.C0 + (p.C1 - p.C0) * _smoothstep(x)


def Phi_ten_of_S(S: np.ndarray, p: BackgroundParams) -> np.ndarray:
    # MVP: phi ~ 0 (N=1). Puedes introducir picos por sellos mas adelante.
    return p.phi_amp * np.zeros_like(S)


def T_of_S(S: np.ndarray, p: BackgroundParams) -> np.ndarray:
    return p.T0 * np.ones_like(S)


def solve_background(S: np.ndarray, p: BackgroundParams):
    """
    Integra desde S4 hacia atras imponiendo normalizacion:
      a(S4)=1, t_rel(S4)=0, H(S4)=H0
    """
    C = C_of_S(S, p)
    Phi = Phi_ten_of_S(S, p)
    N = np.exp(Phi)  # N(S)
    T = T_of_S(S, p)

    a = np.zeros_like(S, dtype=float)
    t = np.zeros_like(S, dtype=float)

    a[-1] = 1.0
    t[-1] = 0.0

    # Integracion hacia atras (S decrece)
    for i in range(len(S) - 2, -1, -1):
        dS = S[i + 1] - S[i]
        # d ln a / dS = C(S)  =>  ln a(S_i) = ln a(S_{i+1}) - C*dS
        a[i] = a[i + 1] * np.exp(-C[i + 1] * dS)
        # dt/dS = T*N
        t[i] = t[i + 1] - (T[i + 1] * N[i + 1]) * dS

    z = 1.0 / a - 1.0

    # En este MVP, escalamos H con C para mantener coherencia con la normalizacion.
    # En la version completa, sustituye por tu Friedmann MCMC (rho_id/rho_lat).
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
    }
