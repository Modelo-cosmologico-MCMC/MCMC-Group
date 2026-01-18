from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoide estable y suave."""
    return 0.5 * (1.0 + np.tanh(x))


def _gauss(S: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussiana normalizada."""
    s2 = max(sigma, 1e-12) ** 2
    return np.exp(-0.5 * (S - mu) ** 2 / s2)


@dataclass(frozen=True)
class CronosShapeParams:
    """
    Parametros para las formas de Cronos refinadas.

    Implementa lo descrito en el tratado computacional:
    - C(S): 3 regimenes con transiciones suaves cerca de S2 y S3
    - T(S): picos en sellos (cronificacion concentrada)
    - Phi_ten(S): envolvente exponencial decreciente + bultos locales en sellos

    Attributes:
        C_early: Valor de C en regimen rigido (S < S2)
        C_mid: Valor de C en regimen intermedio (S2 <= S < S3)
        C_late: Valor de C en regimen casi inercial (S >= S3)
        w_C2: Ancho de transicion alrededor de S2
        w_C3: Ancho de transicion alrededor de S3
        T0: Valor base de T
        t1_amp, t2_amp, t3_amp: Amplitudes de picos en S1, S2, S3
        t_sigma: Ancho de los picos gaussianos de T
        phi_env_amp: Amplitud de la envolvente exponencial
        phi_env_lambda: Tasa de decaimiento de la envolvente
        phi1_amp, phi2_amp, phi3_amp: Amplitudes de bultos en S1, S2, S3
        phi_sigma: Ancho de los bultos gaussianos
    """
    # C(S) - funcion de expansion
    C_early: float = 2.2       # S < S2 (regimen rigido)
    C_mid: float = 1.7         # S2 <= S < S3 (regimen intermedio)
    C_late: float = 1.0        # S >= S3 (regimen casi inercial)
    w_C2: float = 0.015        # ancho transicion alrededor de S2
    w_C3: float = 0.015        # ancho transicion alrededor de S3

    # T(S) - cronificacion
    T0: float = 1.0
    t1_amp: float = 0.20
    t2_amp: float = 0.15
    t3_amp: float = 0.10
    t_sigma: float = 0.010

    # Phi_ten(S) - campo tensional
    phi_env_amp: float = 0.20
    phi_env_lambda: float = 3.0
    phi1_amp: float = 0.20
    phi2_amp: float = 0.15
    phi3_amp: float = 0.10
    phi_sigma: float = 0.010


def C_of_S(S: np.ndarray, p: CronosShapeParams, *, S2: float, S3: float) -> np.ndarray:
    """
    Calcula C(S) con 3 regimenes y transiciones suaves.

    C(S) controla la expansion: d(ln a)/dS = C(S).

    Estructura:
    - Regimen rigido (S < S2): C ~ C_early
    - Regimen intermedio (S2 <= S < S3): C ~ C_mid
    - Regimen casi inercial (S >= S3): C ~ C_late

    Args:
        S: Array de valores de entropia
        p: Parametros de forma
        S2: Segundo sello
        S3: Tercer sello

    Returns:
        Array de C(S)
    """
    S = np.asarray(S, float)

    # Mezcla early -> mid en S2
    x2 = (S - S2) / max(p.w_C2, 1e-12)
    m2 = _sigmoid(x2)  # 0 antes de S2, 1 despues

    # Mezcla mid -> late en S3
    x3 = (S - S3) / max(p.w_C3, 1e-12)
    m3 = _sigmoid(x3)  # 0 antes de S3, 1 despues

    # Composicion
    C_em = (1.0 - m2) * p.C_early + m2 * p.C_mid
    C = (1.0 - m3) * C_em + m3 * p.C_late

    return C


def T_of_S(S: np.ndarray, p: CronosShapeParams, *, S1: float, S2: float, S3: float) -> np.ndarray:
    """
    Calcula T(S) con picos en los sellos.

    T(S) controla la cronificacion: cadencia base del tiempo emergente.
    Presenta picos gaussianos en S1, S2, S3 (cronificacion concentrada
    en los colapsos ontologicos).

    Args:
        S: Array de valores de entropia
        p: Parametros de forma
        S1, S2, S3: Sellos ontologicos

    Returns:
        Array de T(S)
    """
    S = np.asarray(S, float)

    peaks = (
        p.t1_amp * _gauss(S, S1, p.t_sigma)
        + p.t2_amp * _gauss(S, S2, p.t_sigma)
        + p.t3_amp * _gauss(S, S3, p.t_sigma)
    )

    return p.T0 * (1.0 + peaks)


def Phi_ten_of_S(S: np.ndarray, p: CronosShapeParams, *, S1: float, S2: float, S3: float) -> np.ndarray:
    """
    Calcula Phi_ten(S) - Campo de Adrian tensional.

    Combina:
    - Envolvente exponencial decreciente desde S1
    - Bultos gaussianos locales en S1, S2, S3

    Args:
        S: Array de valores de entropia
        p: Parametros de forma
        S1, S2, S3: Sellos ontologicos

    Returns:
        Array de Phi_ten(S)
    """
    S = np.asarray(S, float)

    # Envolvente exponencial decreciente desde S1
    env = p.phi_env_amp * np.exp(-p.phi_env_lambda * np.maximum(S - S1, 0.0))

    # Bultos gaussianos en sellos
    lumps = (
        p.phi1_amp * _gauss(S, S1, p.phi_sigma)
        + p.phi2_amp * _gauss(S, S2, p.phi_sigma)
        + p.phi3_amp * _gauss(S, S3, p.phi_sigma)
    )

    return env + lumps


def N_of_S(S: np.ndarray, p: CronosShapeParams, *, S1: float, S2: float, S3: float) -> np.ndarray:
    """
    Calcula N(S) = exp(Phi_ten(S)) - Lapse entropico.

    N(S) modula el tiempo emergente. Es el exponencial del campo tensional.

    Args:
        S: Array de valores de entropia
        p: Parametros de forma
        S1, S2, S3: Sellos ontologicos

    Returns:
        Array de N(S)
    """
    return np.exp(Phi_ten_of_S(S, p, S1=S1, S2=S2, S3=S3))
