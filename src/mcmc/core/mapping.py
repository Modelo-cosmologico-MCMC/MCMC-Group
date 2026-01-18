"""
Módulo de Mapeo S ↔ z e Interpolaciones
=======================================

Este módulo proporciona herramientas para convertir entre la variable entrópica S
y el redshift z, así como interpolaciones eficientes basadas en splines cúbicos.

El mapeo S ↔ z es fundamental porque:
- El MCMC evoluciona internamente en S
- Los observables y datos están en z
- Necesitamos conversión bidireccional eficiente

Referencias:
    - Tratado MCMC Maestro: correspondencia S ↔ z
    - Apartado Computacional: interpolaciones y splines
"""

from typing import Dict, Tuple, Optional, Union, Callable
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq


class SZMapper:
    """
    Clase para mapeo bidireccional S ↔ z.

    Utiliza interpolación por splines cúbicos para conversiones eficientes
    entre la variable entrópica S y el redshift z.

    Attributes:
        S: Array de valores de S (ordenado creciente)
        z: Array de valores de z correspondientes
        a: Array de factores de escala (opcional)
    """

    def __init__(
        self,
        S: np.ndarray,
        z: np.ndarray,
        a: Optional[np.ndarray] = None,
        extrapolate: bool = False
    ):
        """
        Inicializa el mapper S ↔ z.

        Args:
            S: Array de variable entrópica
            z: Array de redshift
            a: Array de factor de escala (opcional, se calcula de z si no se da)
            extrapolate: Si True, permite extrapolación fuera del rango
        """
        self.S = np.asarray(S)
        self.z = np.asarray(z)
        self.a = np.asarray(a) if a is not None else 1.0 / (1.0 + self.z)

        if len(self.S) != len(self.z):
            raise ValueError("S y z deben tener la misma longitud")

        self._extrapolate = extrapolate
        self._build_splines()

    def _build_splines(self):
        """Construye los splines de interpolación."""
        bc_type = None if self._extrapolate else 'not-a-knot'
        ext_mode = None if self._extrapolate else 'extrapolate'

        # S -> z (S creciente, z generalmente decreciente en late times)
        self._z_of_S = CubicSpline(self.S, self.z, bc_type='not-a-knot')
        self._a_of_S = CubicSpline(self.S, self.a, bc_type='not-a-knot')

        # z -> S (necesitamos ordenar por z creciente)
        idx_z_sorted = np.argsort(self.z)
        z_sorted = self.z[idx_z_sorted]
        S_sorted = self.S[idx_z_sorted]
        a_sorted = self.a[idx_z_sorted]

        self._S_of_z = CubicSpline(z_sorted, S_sorted, bc_type='not-a-knot')
        self._a_of_z_spline = CubicSpline(z_sorted, a_sorted, bc_type='not-a-knot')

        # Guardar rangos válidos
        self._z_min = float(np.min(self.z))
        self._z_max = float(np.max(self.z))
        self._S_min = float(np.min(self.S))
        self._S_max = float(np.max(self.S))

    def z_of_S(self, S: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convierte S → z.

        Args:
            S: Valor(es) de variable entrópica

        Returns:
            Redshift correspondiente
        """
        S = np.atleast_1d(S)
        if not self._extrapolate:
            if np.any(S < self._S_min) or np.any(S > self._S_max):
                raise ValueError(
                    f"S fuera de rango [{self._S_min}, {self._S_max}]"
                )
        result = self._z_of_S(S)
        return float(result) if result.size == 1 else result

    def S_of_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convierte z → S.

        Args:
            z: Valor(es) de redshift

        Returns:
            Variable entrópica correspondiente
        """
        z = np.atleast_1d(z)
        if not self._extrapolate:
            if np.any(z < self._z_min) or np.any(z > self._z_max):
                raise ValueError(
                    f"z fuera de rango [{self._z_min}, {self._z_max}]"
                )
        result = self._S_of_z(z)
        return float(result) if result.size == 1 else result

    def a_of_S(self, S: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte S → a (factor de escala)."""
        S = np.atleast_1d(S)
        result = self._a_of_S(S)
        return float(result) if result.size == 1 else result

    def a_of_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convierte z → a (factor de escala)."""
        return 1.0 / (1.0 + np.atleast_1d(z))

    @property
    def z_range(self) -> Tuple[float, float]:
        """Rango válido de z."""
        return (self._z_min, self._z_max)

    @property
    def S_range(self) -> Tuple[float, float]:
        """Rango válido de S."""
        return (self._S_min, self._S_max)

    def derivative_z_of_S(self, S: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula dz/dS usando el spline."""
        S = np.atleast_1d(S)
        result = self._z_of_S(S, 1)  # Primera derivada
        return float(result) if result.size == 1 else result

    def derivative_S_of_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calcula dS/dz usando el spline."""
        z = np.atleast_1d(z)
        result = self._S_of_z(z, 1)  # Primera derivada
        return float(result) if result.size == 1 else result


class QuantityInterpolator:
    """
    Interpolador genérico para cantidades cosmológicas.

    Permite interpolar cualquier cantidad Q(S) o Q(z) de forma eficiente.
    """

    def __init__(
        self,
        x: np.ndarray,
        Q: np.ndarray,
        kind: str = 'cubic',
        name: str = 'Q'
    ):
        """
        Inicializa el interpolador.

        Args:
            x: Variable independiente (S o z)
            Q: Cantidad a interpolar
            kind: Tipo de interpolación ('linear', 'cubic')
            name: Nombre de la cantidad (para mensajes de error)
        """
        self.x = np.asarray(x)
        self.Q = np.asarray(Q)
        self.name = name
        self.kind = kind

        if kind == 'cubic':
            self._interp = CubicSpline(self.x, self.Q)
        else:
            self._interp = interp1d(self.x, self.Q, kind=kind, fill_value='extrapolate')

        self._x_min = float(np.min(self.x))
        self._x_max = float(np.max(self.x))

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evalúa la cantidad interpolada."""
        x = np.atleast_1d(x)
        result = self._interp(x)
        return float(result) if result.size == 1 else result

    def derivative(self, x: Union[float, np.ndarray], n: int = 1) -> Union[float, np.ndarray]:
        """Calcula la n-ésima derivada."""
        if self.kind != 'cubic':
            raise ValueError("Derivadas solo disponibles para interpolación cúbica")
        x = np.atleast_1d(x)
        result = self._interp(x, n)
        return float(result) if result.size == 1 else result

    @property
    def domain(self) -> Tuple[float, float]:
        """Dominio de interpolación."""
        return (self._x_min, self._x_max)


def create_z_grid(z_min: float = 0.0, z_max: float = 3.0, n_points: int = 1000) -> np.ndarray:
    """
    Crea una rejilla regular en z para observables.

    Args:
        z_min: Redshift mínimo
        z_max: Redshift máximo
        n_points: Número de puntos

    Returns:
        Array de redshifts
    """
    return np.linspace(z_min, z_max, n_points)


def create_log_z_grid(
    z_min: float = 0.01,
    z_max: float = 1100.0,
    n_points: int = 1000
) -> np.ndarray:
    """
    Crea una rejilla logarítmica en z (útil para grandes rangos).

    Args:
        z_min: Redshift mínimo (> 0)
        z_max: Redshift máximo
        n_points: Número de puntos

    Returns:
        Array de redshifts en escala log
    """
    return np.logspace(np.log10(z_min), np.log10(z_max), n_points)


def find_S_at_z(
    z_target: float,
    S: np.ndarray,
    z: np.ndarray,
    tol: float = 1e-8
) -> float:
    """
    Encuentra S correspondiente a un z dado usando bisección.

    Método robusto para casos donde la interpolación no es suficiente.

    Args:
        z_target: Redshift objetivo
        S: Array de variable entrópica
        z: Array de redshift
        tol: Tolerancia

    Returns:
        Valor de S correspondiente a z_target
    """
    # Crear interpolador z(S)
    z_interp = CubicSpline(S, z)

    def objective(s):
        return z_interp(s) - z_target

    # Encontrar raíz por bisección
    try:
        S_result = brentq(objective, S[0], S[-1], xtol=tol)
        return S_result
    except ValueError:
        raise ValueError(f"z_target={z_target} fuera del rango de z: [{z[-1]}, {z[0]}]")


# Exportaciones
__all__ = [
    'SZMapper',
    'QuantityInterpolator',
    'create_z_grid',
    'create_log_z_grid',
    'find_S_at_z',
]
