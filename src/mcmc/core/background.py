"""
Motor de Fondo del MCMC
=======================

Este módulo implementa el núcleo computacional del MCMC para integrar las
ecuaciones de fondo en la variable entrópica S.

El motor produce tablas coherentes de:
    S → {a(S), z(S), t_rel(S), H(S)}

Con normalización en S4: a(S4)=1, t_rel(S4)=0, H(S4)=H0

Las ecuaciones del Bloque 1 son:
    d ln a / dS = C(S)
    dt_rel / dS = T(S) · N(S),    donde N(S) = exp[Φ_ten(S)]

Referencias:
    - Tratado MCMC Maestro, Bloque 1: Núcleo Ontológico Computacional
    - Apartado Computacional: integración y normalización
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from .s_grid import SGrid, Seals, create_default_grid


@dataclass
class BackgroundParams:
    """
    Parámetros del motor de fondo del MCMC.

    Estos parámetros controlan las funciones C(S), T(S) y Φ_ten(S) que
    definen la evolución del fondo cosmológico en el modelo.

    Attributes:
        H0: Constante de Hubble hoy [km/s/Mpc]

        # Parámetros de C(S) - función de expansión
        C_early: Valor de C en régimen temprano (S < S2)
        C_late: Valor de C en régimen tardío (S > S3)
        C_trans_width: Ancho de la transición (en unidades de S)

        # Parámetros de T(S) - función temporal (Ley de Cronos)
        T_base: Valor base de T
        T_peak_S1: Amplitud del pico en S1
        T_peak_S2: Amplitud del pico en S2
        T_peak_S3: Amplitud del pico en S3
        T_peak_width: Ancho de los picos

        # Parámetros de Φ_ten(S) - campo tensional (Campo de Adrián)
        Phi_amplitude: Amplitud de la envolvente
        Phi_decay_rate: Tasa de decaimiento
        Phi_bump_S1: Amplitud del bulto en S1
        Phi_bump_S2: Amplitud del bulto en S2
        Phi_bump_S3: Amplitud del bulto en S3
        Phi_bump_width: Ancho de los bultos
    """
    # Constante de Hubble
    H0: float = 67.4  # km/s/Mpc (Planck 2018)

    # Parámetros C(S) - expansión
    C_early: float = 0.5    # Régimen rígido temprano
    C_late: float = 1.0     # Régimen tardío (normalizado)
    C_trans_center: float = 0.5  # Centro de la transición
    C_trans_width: float = 0.1   # Ancho de la transición

    # Parámetros T(S) - cronificación
    T_base: float = 1.0
    T_peak_S1: float = 0.2   # Intensificación en S1
    T_peak_S2: float = 0.3   # Intensificación en S2
    T_peak_S3: float = 0.1   # Intensificación en S3
    T_peak_width: float = 0.02

    # Parámetros Φ_ten(S) - campo tensional
    Phi_amplitude: float = 0.5     # Amplitud inicial
    Phi_decay_rate: float = 2.0    # Tasa de decaimiento
    Phi_bump_S1: float = 0.3       # Bulto en S1
    Phi_bump_S2: float = 0.2       # Bulto en S2
    Phi_bump_S3: float = 0.1       # Bulto en S3
    Phi_bump_width: float = 0.02

    def to_dict(self) -> Dict[str, float]:
        """Convierte los parámetros a diccionario."""
        return {
            'H0': self.H0,
            'C_early': self.C_early,
            'C_late': self.C_late,
            'C_trans_center': self.C_trans_center,
            'C_trans_width': self.C_trans_width,
            'T_base': self.T_base,
            'T_peak_S1': self.T_peak_S1,
            'T_peak_S2': self.T_peak_S2,
            'T_peak_S3': self.T_peak_S3,
            'T_peak_width': self.T_peak_width,
            'Phi_amplitude': self.Phi_amplitude,
            'Phi_decay_rate': self.Phi_decay_rate,
            'Phi_bump_S1': self.Phi_bump_S1,
            'Phi_bump_S2': self.Phi_bump_S2,
            'Phi_bump_S3': self.Phi_bump_S3,
            'Phi_bump_width': self.Phi_bump_width,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'BackgroundParams':
        """Construye parámetros desde diccionario."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _smooth_step(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Función escalón suave usando tanh.

    Transición de 0 a 1 centrada en `center` con ancho `width`.
    """
    return 0.5 * (1 + np.tanh((x - center) / width))


def _gaussian_bump(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """
    Pulso gaussiano centrado en `center`.
    """
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def C_of_S(S: np.ndarray, p: BackgroundParams, seals: Seals = None) -> np.ndarray:
    """
    Función C(S) que controla la expansión: d ln a / dS = C(S)

    El comportamiento es:
    - Régimen temprano (S < S2): C ≈ C_early (régimen rígido)
    - Régimen tardío (S > S3): C → C_late (normalizado a 1)
    - Transición suave entre ambos regímenes

    Args:
        S: Array de valores de la variable entrópica
        p: Parámetros del fondo
        seals: Sellos ontológicos (opcional)

    Returns:
        Array con C(S)
    """
    if seals is None:
        seals = Seals()

    # Transición suave de C_early a C_late
    transition = _smooth_step(S, p.C_trans_center, p.C_trans_width)
    C = p.C_early + (p.C_late - p.C_early) * transition

    return C


def Phi_ten_of_S(S: np.ndarray, p: BackgroundParams, seals: Seals = None) -> np.ndarray:
    """
    Campo tensional Φ_ten(S) del Campo de Adrián.

    Estructura:
    - Envolvente decreciente exponencial
    - Bultos (bumps) en los sellos S1, S2, S3

    El campo tensional modula la "lapse function" N(S) = exp[Φ_ten(S)]

    Args:
        S: Array de valores de la variable entrópica
        p: Parámetros del fondo
        seals: Sellos ontológicos (opcional)

    Returns:
        Array con Φ_ten(S)
    """
    if seals is None:
        seals = Seals()

    # Envolvente decreciente
    envelope = p.Phi_amplitude * np.exp(-p.Phi_decay_rate * (S - seals.S1))

    # Bultos en los sellos
    bump1 = _gaussian_bump(S, seals.S1, p.Phi_bump_width, p.Phi_bump_S1)
    bump2 = _gaussian_bump(S, seals.S2, p.Phi_bump_width, p.Phi_bump_S2)
    bump3 = _gaussian_bump(S, seals.S3, p.Phi_bump_width, p.Phi_bump_S3)

    Phi = envelope + bump1 + bump2 + bump3

    return Phi


def T_of_S(S: np.ndarray, p: BackgroundParams, seals: Seals = None) -> np.ndarray:
    """
    Función T(S) de cronificación (Ley de Cronos).

    T(S) tiene:
    - Un valor base T_base
    - Picos en los sellos para intensificar la cronificación

    Args:
        S: Array de valores de la variable entrópica
        p: Parámetros del fondo
        seals: Sellos ontológicos (opcional)

    Returns:
        Array con T(S)
    """
    if seals is None:
        seals = Seals()

    # Base más picos en los sellos
    T = np.full_like(S, p.T_base)
    T += _gaussian_bump(S, seals.S1, p.T_peak_width, p.T_peak_S1)
    T += _gaussian_bump(S, seals.S2, p.T_peak_width, p.T_peak_S2)
    T += _gaussian_bump(S, seals.S3, p.T_peak_width, p.T_peak_S3)

    return T


def N_of_S(S: np.ndarray, p: BackgroundParams, seals: Seals = None) -> np.ndarray:
    """
    Función lapse N(S) = exp[Φ_ten(S)].

    Esta es la "función de lapso" que modula el paso temporal.

    Args:
        S: Array de valores de la variable entrópica
        p: Parámetros del fondo
        seals: Sellos ontológicos (opcional)

    Returns:
        Array con N(S)
    """
    Phi = Phi_ten_of_S(S, p, seals)
    return np.exp(Phi)


@dataclass
class BackgroundSolution:
    """
    Solución del fondo cosmológico MCMC.

    Contiene todas las cantidades de fondo integradas sobre la rejilla S.
    """
    S: np.ndarray           # Variable entrópica
    a: np.ndarray           # Factor de escala
    z: np.ndarray           # Redshift
    t_rel: np.ndarray       # Tiempo relativo (t_rel=0 hoy)
    H: np.ndarray           # Parámetro de Hubble
    C: np.ndarray           # Función C(S)
    T: np.ndarray           # Función T(S)
    N: np.ndarray           # Función lapse N(S)
    Phi_ten: np.ndarray     # Campo tensional
    params: BackgroundParams

    def __post_init__(self):
        """Construye interpoladores."""
        self._build_interpolators()

    def _build_interpolators(self):
        """Construye splines cúbicos para interpolación."""
        # Interpoladores S -> cantidades
        self._a_of_S = CubicSpline(self.S, self.a)
        self._z_of_S = CubicSpline(self.S, self.z)
        self._H_of_S = CubicSpline(self.S, self.H)
        self._t_of_S = CubicSpline(self.S, self.t_rel)

        # Interpolador z -> S (invertido, S creciente con z decreciente)
        # Necesitamos ordenar por z creciente
        idx_sorted = np.argsort(self.z)
        self._S_of_z = CubicSpline(self.z[idx_sorted], self.S[idx_sorted])
        self._a_of_z = CubicSpline(self.z[idx_sorted], self.a[idx_sorted])
        self._H_of_z = CubicSpline(self.z[idx_sorted], self.H[idx_sorted])

    def get_at_z(self, z: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Obtiene cantidades de fondo a redshift z dado.

        Args:
            z: Redshift(s) donde evaluar

        Returns:
            Diccionario con a, H, S, t_rel interpolados
        """
        z = np.atleast_1d(z)
        return {
            'z': z,
            'S': self._S_of_z(z),
            'a': self._a_of_z(z),
            'H': self._H_of_z(z),
        }

    def H_at_z(self, z: float) -> float:
        """Retorna H(z) interpolado."""
        return float(self._H_of_z(z))

    def a_at_z(self, z: float) -> float:
        """Retorna a(z) interpolado."""
        return float(self._a_of_z(z))


def solve_background(
    S: np.ndarray,
    p: BackgroundParams,
    seals: Seals = None,
    validate: bool = True
) -> BackgroundSolution:
    """
    Integra las ecuaciones de fondo del MCMC.

    Las ecuaciones son (Bloque 1):
        d ln a / dS = C(S)
        dt_rel / dS = T(S) · N(S),    donde N(S) = exp[Φ_ten(S)]

    Con condiciones de normalización en S4:
        a(S4) = 1
        t_rel(S4) = 0
        H(S4) = H0

    La integración se realiza hacia atrás desde S4.

    Args:
        S: Array con la rejilla entrópica
        p: Parámetros del fondo
        seals: Sellos ontológicos (usa defaults si None)
        validate: Si True, valida la solución

    Returns:
        BackgroundSolution con todas las cantidades de fondo
    """
    if seals is None:
        seals = Seals()

    # Calcular funciones auxiliares
    C = C_of_S(S, p, seals)
    Phi = Phi_ten_of_S(S, p, seals)
    N = np.exp(Phi)
    T = T_of_S(S, p, seals)

    # Arrays de salida
    a = np.zeros_like(S)
    t_rel = np.zeros_like(S)

    # Condiciones en S4 (último punto)
    a[-1] = 1.0
    t_rel[-1] = 0.0

    # Integración hacia atrás (desde S4 hacia S1)
    # d ln a / dS = C  =>  a[i] = a[i+1] * exp(-C[i+1] * dS)
    # dt / dS = T*N    =>  t[i] = t[i+1] - T[i+1]*N[i+1] * dS
    for i in range(len(S) - 2, -1, -1):
        dS = S[i + 1] - S[i]
        # Usamos el valor en i+1 para el paso (Euler hacia atrás)
        a[i] = a[i + 1] * np.exp(-C[i + 1] * dS)
        t_rel[i] = t_rel[i + 1] - (T[i + 1] * N[i + 1]) * dS

    # Redshift
    z = 1.0 / a - 1.0

    # Parámetro de Hubble: H(S) proporcional a C(S), normalizado en S4
    # H(S4) = H0, y H escala como C/C(S4)
    H = p.H0 * (C / C[-1])

    # Crear solución
    sol = BackgroundSolution(
        S=S, a=a, z=z, t_rel=t_rel, H=H,
        C=C, T=T, N=N, Phi_ten=Phi,
        params=p
    )

    # Validación ontológica
    if validate:
        _validate_background(sol, seals)

    return sol


def _validate_background(sol: BackgroundSolution, seals: Seals) -> None:
    """
    Validación ontológica del fondo (Bloque 1).

    Verifica:
    1. Monotonía de a(S): debe ser creciente
    2. Positividad: a > 0, H > 0 en todo el rango
    3. Normalización: a(S4)=1, H(S4)=H0
    """
    # 1. Monotonía
    da = np.diff(sol.a)
    if not np.all(da > 0):
        n_violations = np.sum(da <= 0)
        raise ValueError(
            f"Violación de monotonía: a(S) no es estrictamente creciente. "
            f"{n_violations} puntos con da/dS <= 0"
        )

    # 2. Positividad
    if not np.all(sol.a > 0):
        raise ValueError("Violación de positividad: a(S) tiene valores no positivos")

    if not np.all(sol.H > 0):
        raise ValueError("Violación de positividad: H(S) tiene valores no positivos")

    # 3. Normalización en S4
    if not np.isclose(sol.a[-1], 1.0, atol=1e-10):
        raise ValueError(f"Normalización incorrecta: a(S4) = {sol.a[-1]} != 1")

    if not np.isclose(sol.H[-1], sol.params.H0, atol=1e-6):
        raise ValueError(
            f"Normalización incorrecta: H(S4) = {sol.H[-1]} != H0 = {sol.params.H0}"
        )


def solve_background_default() -> BackgroundSolution:
    """
    Función de conveniencia para resolver el fondo con parámetros por defecto.

    Returns:
        BackgroundSolution con la solución completa
    """
    grid, S = create_default_grid()
    params = BackgroundParams()
    return solve_background(S, params, grid.seals)


# Exportaciones principales
__all__ = [
    'BackgroundParams',
    'BackgroundSolution',
    'solve_background',
    'solve_background_default',
    'C_of_S',
    'T_of_S',
    'N_of_S',
    'Phi_ten_of_S',
]
