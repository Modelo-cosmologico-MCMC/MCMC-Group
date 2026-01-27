"""Mapa Entrópico Completo S↔z↔t↔a.

Este módulo implementa el puente fundamental que conecta la ontología
tensional (S) con las variables observacionales (z, t, a).

Ecuaciones canónicas:
    Ley de Cronos: dt_rel/dS = T(S)·N(S), donde N(S) = exp[Φ_ten(S)]
    Mapa S(z): S(z) = S_BB + (S_0 - S_BB)·(1+z)^(-p)
    Factor de escala: a(S) = 1/(1+z(S))

Relaciones:
    S = 1.001 (S_BB) → z = ∞ (Big Bang)
    S = S_0 ≈ 1.0015 → z = 0 (hoy)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import quad



@dataclass
class SMapParams:
    """Parámetros del mapa S↔z↔t↔a.

    Attributes:
        S_BB: Big Bang observable (umbral 4º colapso)
        S_0: S actual (hoy, z=0)
        p: Exponente de transición en S(z)
        lambda_C: Parámetro base de la Ley de Cronos
        k_alpha: Normalización temporal
    """
    S_BB: float = 1.001
    S_0: float = 1.0015
    p: float = 2.0
    lambda_C: float = 0.1
    k_alpha: float = 1.0


class EntropyMap:
    """Mapa completo S↔z↔t↔a con Ley de Cronos.

    Proporciona conversiones bidireccionales entre:
    - Variable entrópica S
    - Redshift z
    - Tiempo cósmico t (anclado: t=0 en Big Bang)
    - Factor de escala a

    Attributes:
        params: Parámetros del mapa
        phi_ten: Función Φ_ten(S) para el lapse N(S)
    """

    def __init__(
        self,
        params: SMapParams | None = None,
        phi_ten_func: Callable[[float], float] | None = None
    ):
        """Inicializa el mapa entrópico.

        Args:
            params: Parámetros del mapa (defaults si None)
            phi_ten_func: Función Φ_ten(S) para lapse (0 si None → N=1)
        """
        self.params = params or SMapParams()
        self._phi_ten = phi_ten_func or (lambda S: 0.0)
        self._z_interp = None
        self._t_interp = None

    # -------------------------------------------------------------------------
    # Funciones de la Ley de Cronos
    # -------------------------------------------------------------------------

    def C_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """C(S) = d ln a / dS - tasa de expansión entrópica.

        Parametrización: C(S) = λ_C · tanh(S/λ_C)

        Args:
            S: Variable entrópica

        Returns:
            Tasa de expansión entrópica
        """
        p = self.params
        S_arr = np.asarray(S)
        arg = S_arr / max(p.lambda_C, 1e-12)
        return p.lambda_C * np.tanh(arg)

    def T_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """T(S) - función de cronificación base.

        T(S) = λ_C / k_α (forma simple)

        Args:
            S: Variable entrópica

        Returns:
            Función de cronificación
        """
        return self.params.lambda_C / self.params.k_alpha

    def phi_ten(self, S: float | np.ndarray) -> float | np.ndarray:
        """Φ_ten(S) - faz tensorial del Campo de Adrián.

        Args:
            S: Variable entrópica

        Returns:
            Valor de Φ_ten(S)
        """
        S_arr = np.atleast_1d(S)
        result = np.array([self._phi_ten(float(s)) for s in S_arr])
        return result[0] if np.ndim(S) == 0 else result

    def N_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """N(S) = exp[Φ_ten(S)] - lapse entrópico.

        El lapse modula la componente temporal de la métrica:
        g_tt(S) = -N²(S)

        En el límite ΛCDM: Φ_ten → 0 ⇒ N → 1

        Args:
            S: Variable entrópica

        Returns:
            Lapse entrópico
        """
        return np.exp(self.phi_ten(S))

    def dt_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """dt_rel/dS según Ley de Cronos.

        dt_rel/dS = T(S) · N(S)

        Args:
            S: Variable entrópica

        Returns:
            Derivada dt/dS
        """
        return self.T_of_S(S) * self.N_of_S(S)

    # -------------------------------------------------------------------------
    # Mapeo S ↔ z
    # -------------------------------------------------------------------------

    def S_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """S(z) según parametrización canónica.

        S(z) = S_BB + (S_0 - S_BB) · (1+z)^(-p)

        Propiedades:
        - S(z=0) = S_0 (hoy)
        - S(z→∞) → S_BB (Big Bang)
        - dS/dz < 0 (monótona decreciente)

        Args:
            z: Redshift

        Returns:
            Variable entrópica S
        """
        p = self.params
        z_arr = np.asarray(z)

        delta_S = p.S_0 - p.S_BB
        return p.S_BB + delta_S * (1.0 + z_arr) ** (-p.p)

    def z_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """z(S) - inversión del mapa S(z).

        z(S) = [(S - S_BB)/(S_0 - S_BB)]^(-1/p) - 1

        Args:
            S: Variable entrópica

        Returns:
            Redshift z
        """
        p = self.params
        S_arr = np.asarray(S)

        delta_S = p.S_0 - p.S_BB
        ratio = np.maximum((S_arr - p.S_BB) / delta_S, 1e-30)

        return ratio ** (-1.0 / p.p) - 1.0

    def dS_dz(self, z: float | np.ndarray) -> float | np.ndarray:
        """dS/dz - derivada del mapa S(z).

        dS/dz = -p · (S_0 - S_BB) · (1+z)^(-p-1)

        Args:
            z: Redshift

        Returns:
            Derivada dS/dz (negativa)
        """
        p = self.params
        z_arr = np.asarray(z)

        delta_S = p.S_0 - p.S_BB
        return -p.p * delta_S * (1.0 + z_arr) ** (-p.p - 1.0)

    # -------------------------------------------------------------------------
    # Mapeo S ↔ t
    # -------------------------------------------------------------------------

    def t_rel_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Tiempo relativo integrado desde S_BB.

        t_rel(S) = ∫_{S_BB}^{S} T(S')·N(S') dS'

        Convención: t_rel(S_BB) = 0 (Big Bang)

        Args:
            S: Variable entrópica

        Returns:
            Tiempo relativo (t=0 en Big Bang)
        """
        S_arr = np.atleast_1d(S)
        results = np.zeros_like(S_arr, dtype=float)

        for i, s in enumerate(S_arr):
            if s <= self.params.S_BB:
                # Pre-Big-Bang: t < 0
                integral, _ = quad(self.dt_dS, s, self.params.S_BB)
                results[i] = -integral
            else:
                # Post-Big-Bang: t > 0
                integral, _ = quad(self.dt_dS, self.params.S_BB, s)
                results[i] = integral

        return results[0] if np.ndim(S) == 0 else results

    def S_of_t(
        self,
        t: float | np.ndarray,
        bracket: tuple[float, float] = (0.9, 1.5),
        n_iter: int = 60
    ) -> float | np.ndarray:
        """S(t) - inversión del mapa t(S) por bisección.

        Args:
            t: Tiempo (t=0 en Big Bang)
            bracket: Rango de búsqueda (S_min, S_max)
            n_iter: Iteraciones de bisección

        Returns:
            Variable entrópica S
        """
        t_arr = np.atleast_1d(t)
        results = np.zeros_like(t_arr, dtype=float)

        for i, ti in enumerate(t_arr):
            lo, hi = bracket

            # Expandir bracket si necesario
            for _ in range(20):
                t_lo = float(self.t_rel_of_S(lo))
                t_hi = float(self.t_rel_of_S(hi))
                if t_lo <= ti <= t_hi:
                    break
                if ti < t_lo:
                    lo = max(lo - 0.1, 0.001)
                if ti > t_hi:
                    hi = hi + 0.5

            # Bisección
            for _ in range(n_iter):
                mid = 0.5 * (lo + hi)
                t_mid = float(self.t_rel_of_S(mid))
                if t_mid < ti:
                    lo = mid
                else:
                    hi = mid

            results[i] = 0.5 * (lo + hi)

        return results[0] if np.ndim(t) == 0 else results

    # -------------------------------------------------------------------------
    # Mapeo S ↔ a
    # -------------------------------------------------------------------------

    def a_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Factor de escala a(S) = 1/(1+z(S)).

        Args:
            S: Variable entrópica

        Returns:
            Factor de escala
        """
        z = self.z_of_S(S)
        return 1.0 / (1.0 + z)

    def S_of_a(self, a: float | np.ndarray) -> float | np.ndarray:
        """S(a) vía z = 1/a - 1.

        Args:
            a: Factor de escala

        Returns:
            Variable entrópica S
        """
        a_arr = np.asarray(a)
        z = 1.0 / a_arr - 1.0
        return self.S_of_z(z)

    def dlna_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """d ln a / dS = C(S).

        Args:
            S: Variable entrópica

        Returns:
            Derivada d ln a / dS
        """
        return self.C_of_S(S)

    # -------------------------------------------------------------------------
    # Verificaciones y límites
    # -------------------------------------------------------------------------

    def is_LCDM_limit(self, tol: float = 1e-6) -> bool:
        """Verifica si estamos en el límite ΛCDM (Φ_ten ≈ 0).

        Args:
            tol: Tolerancia

        Returns:
            True si Φ_ten es despreciable
        """
        S_test = np.linspace(self.params.S_BB, self.params.S_0, 10)
        phi_vals = self.phi_ten(S_test)
        return np.all(np.abs(phi_vals) < tol)

    def verify_monotonicity(self) -> tuple[bool, str]:
        """Verifica monotonicidad de los mapeos.

        Returns:
            (is_valid, message)
        """
        z_test = np.linspace(0, 10, 100)
        S_test = self.S_of_z(z_test)

        # dS/dz debe ser negativo
        dS = np.diff(S_test)
        dz = np.diff(z_test)
        dS_dz_numerical = dS / dz

        if not np.all(dS_dz_numerical < 0):
            return False, "S(z) no es monótona decreciente"

        # t(S) debe ser creciente
        S_range = np.linspace(self.params.S_BB - 0.01, self.params.S_0 + 0.01, 50)
        t_test = self.t_rel_of_S(S_range)
        dt = np.diff(t_test)

        if not np.all(dt > 0):
            return False, "t(S) no es monótona creciente"

        return True, "OK: todos los mapeos son monótonos"


# =============================================================================
# Funciones de conveniencia
# =============================================================================

def create_default_map(
    phi_ten_func: Callable[[float], float] | None = None
) -> EntropyMap:
    """Crea un EntropyMap con parámetros por defecto.

    Args:
        phi_ten_func: Función Φ_ten(S) opcional

    Returns:
        EntropyMap configurado
    """
    return EntropyMap(SMapParams(), phi_ten_func)


def S_of_z_simple(z: np.ndarray, S_BB: float = 1.001, S_0: float = 1.0015, p: float = 2.0) -> np.ndarray:
    """Versión simplificada de S(z) sin crear objeto.

    Args:
        z: Redshift array
        S_BB: Big Bang threshold
        S_0: S actual
        p: Exponente

    Returns:
        S array
    """
    return S_BB + (S_0 - S_BB) * (1.0 + z) ** (-p)


def z_of_S_simple(S: np.ndarray, S_BB: float = 1.001, S_0: float = 1.0015, p: float = 2.0) -> np.ndarray:
    """Versión simplificada de z(S) sin crear objeto.

    Args:
        S: Entropic variable array
        S_BB: Big Bang threshold
        S_0: S actual
        p: Exponente

    Returns:
        z array
    """
    ratio = np.maximum((S - S_BB) / (S_0 - S_BB), 1e-30)
    return ratio ** (-1.0 / p) - 1.0
