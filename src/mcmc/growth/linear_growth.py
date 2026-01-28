"""Crecimiento Lineal D(S) en Variable Entrópica.

En el MCMC, las ecuaciones de crecimiento se reformulan en términos de S.

Ecuación de crecimiento estándar:
    D''(a) + [2 + d ln H/d ln a] D'(a) - (3/2) Ω_m(a) D(a) = 0

Transformación a variable S:
    D̈(S) + Ξ(S) Ḋ(S) - (3/2) G(S) Ω_m(S) C²(S) D(S) = 0

donde:
    Ξ(S) = C(S)[2 + d ln H/d ln a] + C'(S)/C(S)  [fricción efectiva]
    G(S) = 1 + ΔG(S; Φ_ten, ρ_lat, ...)         [gravedad efectiva]
    C(S) = d ln a / dS                           [Ley de Cronos]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import odeint


@dataclass
class GrowthParams:
    """Parámetros para el crecimiento lineal.

    Attributes:
        Omega_m0: Fracción de materia hoy
        Omega_Lambda0: Fracción de energía oscura hoy
        H0: Constante de Hubble km/s/Mpc
        sigma8_0: σ₈ normalizado a z=0
        gamma_growth: Exponente de crecimiento (≈0.55 para ΛCDM)
    """
    Omega_m0: float = 0.315
    Omega_Lambda0: float = 0.685
    H0: float = 67.4
    sigma8_0: float = 0.811
    gamma_growth: float = 0.55


class LinearGrowthSolver:
    """Resuelve las ecuaciones de crecimiento en variable S.

    Attributes:
        params: Parámetros de crecimiento
        S_map: Mapa entrópico (opcional)
        G_eff_func: Función G(S) de gravedad efectiva
    """

    def __init__(
        self,
        params: GrowthParams | None = None,
        S_of_z: Callable[[np.ndarray], np.ndarray] | None = None,
        z_of_S: Callable[[np.ndarray], np.ndarray] | None = None,
        C_of_S: Callable[[np.ndarray], np.ndarray] | None = None,
        G_eff: Callable[[float], float] | None = None
    ):
        """Inicializa el solver.

        Args:
            params: Parámetros de crecimiento
            S_of_z: Mapeo S(z)
            z_of_S: Mapeo z(S)
            C_of_S: Función C(S) = d ln a / dS
            G_eff: Gravedad efectiva G(S)
        """
        self.params = params or GrowthParams()

        # Mapeos por defecto (simple)
        self._S_of_z = S_of_z or self._default_S_of_z
        self._z_of_S = z_of_S or self._default_z_of_S
        self._C_of_S = C_of_S or self._default_C_of_S
        self._G_eff = G_eff or (lambda S: 1.0)

    def _default_S_of_z(self, z: np.ndarray) -> np.ndarray:
        """Mapeo S(z) por defecto."""
        S_BB = 1.001
        S_0 = 1.0015
        p = 2.0
        return S_BB + (S_0 - S_BB) * (1.0 + np.asarray(z)) ** (-p)

    def _default_z_of_S(self, S: np.ndarray) -> np.ndarray:
        """Mapeo z(S) por defecto."""
        S_BB = 1.001
        S_0 = 1.0015
        p = 2.0
        S_arr = np.asarray(S)
        ratio = np.maximum((S_arr - S_BB) / (S_0 - S_BB), 1e-30)
        return ratio ** (-1.0 / p) - 1.0

    def _default_C_of_S(self, S: np.ndarray) -> np.ndarray:
        """C(S) por defecto."""
        lambda_C = 0.1
        S_arr = np.asarray(S)
        return lambda_C * np.tanh(S_arr / lambda_C)

    def H_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """Parámetro de Hubble H(z).

        H²(z) = H₀² [Ω_m(1+z)³ + Ω_Λ]

        Args:
            z: Redshift

        Returns:
            H(z) en km/s/Mpc
        """
        p = self.params
        z_arr = np.asarray(z)
        E2 = p.Omega_m0 * (1.0 + z_arr)**3 + p.Omega_Lambda0
        return p.H0 * np.sqrt(E2)

    def Omega_m_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """Ω_m(z) = Ω_m0 (1+z)³ / E²(z).

        Args:
            z: Redshift

        Returns:
            Fracción de materia
        """
        p = self.params
        z_arr = np.asarray(z)
        E2 = p.Omega_m0 * (1.0 + z_arr)**3 + p.Omega_Lambda0
        return p.Omega_m0 * (1.0 + z_arr)**3 / E2

    def Xi_of_S(self, S: float) -> float:
        """Fricción efectiva Ξ(S).

        Ξ(S) = C(S)[2 + d ln H/d ln a] + dC/dS / C(S)

        Args:
            S: Variable entrópica

        Returns:
            Fricción efectiva
        """
        z = float(self._z_of_S(np.array([S]))[0])
        C = float(self._C_of_S(np.array([S]))[0])

        # d ln H / d ln a (numérico)
        dS = 1e-5
        z_plus = float(self._z_of_S(np.array([S + dS]))[0])
        H = self.H_of_z(z)
        H_plus = self.H_of_z(z_plus)

        # d ln a = C dS
        dlna = C * dS
        if abs(dlna) < 1e-15:
            dlnH_dlna = 0.0
        else:
            dlnH_dlna = (np.log(H_plus) - np.log(H)) / dlna

        # dC/dS
        C_plus = float(self._C_of_S(np.array([S + dS]))[0])
        dC_dS = (C_plus - C) / dS

        C_safe = max(abs(C), 1e-15)
        return C * (2.0 + dlnH_dlna) + dC_dS / C_safe

    def growth_ode_S(self, y: np.ndarray, S: float) -> np.ndarray:
        """ODE del crecimiento en variable S.

        Sistema: y = [D, dD/dS]

        D̈ = -Ξ(S)·Ḋ + (3/2)·G(S)·Ω_m(S)·C²(S)·D

        Args:
            y: Estado [D, dD/dS]
            S: Variable entrópica

        Returns:
            Derivada [dD/dS, d²D/dS²]
        """
        D, dD_dS = y

        z = float(self._z_of_S(np.array([S]))[0])
        C = float(self._C_of_S(np.array([S]))[0])
        Xi = self.Xi_of_S(S)
        G = self._G_eff(S)
        Omega_m = float(self.Omega_m_of_z(z))

        # d²D/dS² = -Ξ·dD/dS + (3/2)·G·Ω_m·C²·D
        d2D_dS2 = -Xi * dD_dS + 1.5 * G * Omega_m * C**2 * D

        return np.array([dD_dS, d2D_dS2])

    def solve_growth_S(
        self,
        S_range: tuple[float, float] = (1.001, 1.0015),
        n_points: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resuelve D(S) en el rango dado.

        Args:
            S_range: (S_min, S_max)
            n_points: Número de puntos

        Returns:
            (S_array, D_array) normalizado D(S_max)=1
        """
        S_arr = np.linspace(S_range[0], S_range[1], n_points)

        # Condiciones iniciales: D pequeño, en modo creciente
        y0 = [1e-5, 1e-5]

        sol = odeint(self.growth_ode_S, y0, S_arr)
        D_arr = sol[:, 0]

        # Normalizar D(S_0) = 1
        D_arr = D_arr / D_arr[-1]

        return S_arr, D_arr

    def solve_growth_z(
        self,
        z_range: tuple[float, float] = (0, 10),
        n_points: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resuelve D(z) en el rango dado.

        Args:
            z_range: (z_min, z_max)
            n_points: Número de puntos

        Returns:
            (z_array, D_array) normalizado D(z=0)=1
        """
        z_arr = np.linspace(z_range[0], z_range[1], n_points)

        # Convertir a S
        S_arr = self._S_of_z(z_arr)

        # Resolver en S
        S_range = (float(S_arr[-1]), float(S_arr[0]))  # S decrece con z
        S_sol, D_sol = self.solve_growth_S(S_range, n_points)

        # Interpolar a z
        D_arr = np.interp(S_arr, S_sol, D_sol)

        # Normalizar D(z=0)=1
        D_arr = D_arr / D_arr[0]

        return z_arr, D_arr

    def f_of_z(self, z: np.ndarray) -> np.ndarray:
        """Tasa de crecimiento f(z) = d ln D / d ln a.

        Args:
            z: Array de redshifts

        Returns:
            f(z)
        """
        z_arr, D_arr = self.solve_growth_z((0, float(np.max(z)) * 1.1))

        # f = d ln D / d ln a = -d ln D / d ln(1+z) = -(1+z)/D · dD/dz
        dD_dz = np.gradient(D_arr, z_arr)
        f_arr = -(1.0 + z_arr) * dD_dz / np.maximum(D_arr, 1e-30)

        # Interpolar a z deseado
        return np.interp(z, z_arr, f_arr)


# =============================================================================
# Funciones de referencia ΛCDM
# =============================================================================

def D_of_z_LCDM(
    z: np.ndarray,
    Omega_m0: float = 0.315,
    Omega_Lambda0: float = 0.685
) -> np.ndarray:
    """Factor de crecimiento D(z) para ΛCDM (aproximación de Carroll).

    D(a) ∝ g(a) / g(1) donde g es la función de crecimiento integral.

    Aproximación de Carroll et al. (1992):
    D(a) ≈ a · [Ω_m(a)]^0.55 / [Ω_m^0.55 + Ω_Λ·(1 + Ω_m/2)]

    Args:
        z: Array de redshifts
        Omega_m0: Fracción de materia
        Omega_Lambda0: Fracción de energía oscura

    Returns:
        D(z) normalizado a D(z=0)=1
    """
    z_arr = np.asarray(z)
    a = 1.0 / (1.0 + z_arr)

    # Ω_m(a)
    E2 = Omega_m0 / a**3 + Omega_Lambda0
    Omega_m_a = (Omega_m0 / a**3) / E2

    # Aproximación de Carroll
    g = (5.0 / 2.0) * Omega_m_a / (
        Omega_m_a**(4.0/7.0)
        - Omega_Lambda0
        + (1.0 + Omega_m_a/2.0) * (1.0 + Omega_Lambda0/70.0)
    )

    # Normalizar a z=0
    g_0 = (5.0 / 2.0) * Omega_m0 / (
        Omega_m0**(4.0/7.0)
        - Omega_Lambda0
        + (1.0 + Omega_m0/2.0) * (1.0 + Omega_Lambda0/70.0)
    )

    return (a * g) / g_0


def f_of_z_LCDM(
    z: np.ndarray,
    Omega_m0: float = 0.315,
    gamma: float = 0.55
) -> np.ndarray:
    """Tasa de crecimiento f(z) ≈ Ω_m(z)^γ para ΛCDM.

    Args:
        z: Array de redshifts
        Omega_m0: Fracción de materia
        gamma: Exponente de crecimiento (≈0.55)

    Returns:
        f(z)
    """
    z_arr = np.asarray(z)
    Omega_Lambda0 = 1.0 - Omega_m0

    E2 = Omega_m0 * (1.0 + z_arr)**3 + Omega_Lambda0
    Omega_m_z = Omega_m0 * (1.0 + z_arr)**3 / E2

    return Omega_m_z ** gamma
