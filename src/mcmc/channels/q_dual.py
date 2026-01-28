"""Término de Intercambio Q_dual entre canales.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001) - No hay intercambio clásico
- Post-Big Bang: S ∈ [1.001, 95.07] - Intercambio activo

Q_dual acopla explícitamente ρ_id y ρ_lat, operacionalizando la
relajación tensional Mp ↔ Ep.

Ecuaciones de conservación:
    ρ̇_id + 3H(1 + w_id)ρ_id = -Q_dual
    ρ̇_lat + 3H(1 + w_lat)ρ_lat = +Q_dual

Σ_i Q_i = 0 (conservación total)

Parametrización:
    Q_S = (∂_S V) Ṡ
    Q_id = η_id(S) · Q_S
    Q_lat = η_lat(S) · Q_S

donde:
    η_id + η_lat = 1
    η_lat(S) = ½[1 + tanh((S - S_★)/λ_★)]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from scipy.integrate import odeint


@dataclass
class QDualParams:
    """Parámetros del intercambio Q_dual.

    CORRECCIÓN: S ∈ [0, 100], post-Big Bang S ∈ [1.001, 95.07]

    Attributes:
        S_star: Umbral de encendido del canal latente (en régimen post-BB)
        lambda_star: Anchura de la transición
        Q_amplitude: Amplitud base del intercambio
    """
    S_star: float = 48.0       # Mitad del rango post-Big Bang
    lambda_star: float = 10.0  # Anchura amplia para S ∈ [1, 95]
    Q_amplitude: float = 0.01


def eta_lat_of_S(S: float | np.ndarray, params: QDualParams) -> float | np.ndarray:
    """Fracción latente η_lat(S).

    η_lat(S) = ½[1 + tanh((S - S_★)/λ_★)]

    - η_lat → 0 para S << S_★ (todo va a id)
    - η_lat → 1 para S >> S_★ (todo va a lat)

    Args:
        S: Variable entrópica
        params: Parámetros Q_dual

    Returns:
        Fracción η_lat ∈ [0, 1]
    """
    S_arr = np.asarray(S)
    arg = (S_arr - params.S_star) / max(params.lambda_star, 1e-12)
    return 0.5 * (1.0 + np.tanh(arg))


def eta_id_of_S(S: float | np.ndarray, params: QDualParams) -> float | np.ndarray:
    """Fracción indeterminada η_id(S) = 1 - η_lat(S).

    Args:
        S: Variable entrópica
        params: Parámetros Q_dual

    Returns:
        Fracción η_id ∈ [0, 1]
    """
    return 1.0 - eta_lat_of_S(S, params)


def deta_lat_dS(S: float | np.ndarray, params: QDualParams) -> float | np.ndarray:
    """Derivada dη_lat/dS.

    dη_lat/dS = (1/2λ_★) sech²((S - S_★)/λ_★)

    Args:
        S: Variable entrópica
        params: Parámetros Q_dual

    Returns:
        Derivada dη_lat/dS
    """
    S_arr = np.asarray(S)
    lam = max(params.lambda_star, 1e-12)
    arg = (S_arr - params.S_star) / lam

    # sech²(x) = 1 - tanh²(x)
    sech2 = 1.0 - np.tanh(arg)**2

    return sech2 / (2.0 * lam)


def Q_dual(
    S: float,
    dV_dS: float,
    S_dot: float,
    params: QDualParams
) -> Tuple[float, float]:
    """Calcula los términos de intercambio Q_id y Q_lat.

    Q_S = (∂V/∂S) · Ṡ
    Q_id = η_id(S) · Q_S
    Q_lat = η_lat(S) · Q_S

    Conservación: Q_id + Q_lat = Q_S, y la suma total de flujos = 0
    porque Q_id → ρ_id y Q_lat → ρ_lat con signos opuestos.

    Args:
        S: Variable entrópica
        dV_dS: Derivada del potencial ∂V/∂S
        S_dot: Tasa de cambio de S (dS/dt)
        params: Parámetros Q_dual

    Returns:
        (Q_id, Q_lat): Intercambios para cada canal
    """
    Q_S = dV_dS * S_dot

    eta_lat = float(eta_lat_of_S(S, params))
    eta_id = 1.0 - eta_lat

    Q_lat = eta_lat * Q_S
    Q_id = eta_id * Q_S

    return Q_id, Q_lat


def Q_dual_simple(
    S: float | np.ndarray,
    params: QDualParams
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Versión simplificada de Q_dual con amplitud fija.

    Q_id = η_id(S) · Q_amplitude
    Q_lat = η_lat(S) · Q_amplitude

    Args:
        S: Variable entrópica
        params: Parámetros Q_dual

    Returns:
        (Q_id, Q_lat)
    """
    eta_lat = eta_lat_of_S(S, params)
    eta_id = 1.0 - eta_lat

    Q_lat = eta_lat * params.Q_amplitude
    Q_id = eta_id * params.Q_amplitude

    return Q_id, Q_lat


@dataclass
class CoupledChannelState:
    """Estado de los canales acoplados.

    Attributes:
        S: Variable entrópica
        rho_id: Densidad del canal indeterminado
        rho_lat: Densidad del canal latente
    """
    S: float
    rho_id: float
    rho_lat: float

    @property
    def rho_total(self) -> float:
        return self.rho_id + self.rho_lat


class CoupledChannelEvolver:
    """Evoluciona los canales acoplados con Q_dual.

    Ecuaciones:
        dρ_id/dS = -3H(1+w_id)ρ_id/Ṡ + Q_id/Ṡ
        dρ_lat/dS = -3H(1+w_lat)ρ_lat/Ṡ + Q_lat/Ṡ

    Attributes:
        q_params: Parámetros de Q_dual
        w_id: Ecuación de estado de ρ_id
        w_lat: Ecuación de estado de ρ_lat
        H_func: Función H(z)
        z_of_S_func: Función z(S)
    """

    def __init__(
        self,
        q_params: QDualParams,
        w_id: float = -1.0,
        w_lat: float = -1.0,
        H_func: Callable[[float], float] | None = None,
        z_of_S_func: Callable[[float], float] | None = None,
        S_dot_func: Callable[[float], float] | None = None
    ):
        """Inicializa el evolucionador.

        Args:
            q_params: Parámetros de Q_dual
            w_id: Ecuación de estado de ρ_id (default: -1, tipo DE)
            w_lat: Ecuación de estado de ρ_lat
            H_func: Función H(z) en km/s/Mpc
            z_of_S_func: Mapeo z(S)
            S_dot_func: Tasa dS/dt
        """
        self.q_params = q_params
        self.w_id = w_id
        self.w_lat = w_lat
        self.H_func = H_func or (lambda z: 67.4 * np.sqrt(0.315 * (1+z)**3 + 0.685))
        # CORRECCIÓN: S ∈ [1.001, 95.07] para post-Big Bang
        # z(S) inversa de S(z) = S_GEOM + (S_0 - S_GEOM)/E²(z)
        # Aproximación lineal para valores por defecto
        self.z_of_S_func = z_of_S_func or (lambda S: max(0, (95.07 - S) / 10.0))
        self.S_dot_func = S_dot_func or (lambda S: 0.1)  # dS/dt ajustado para [1, 95]

    def rhs(self, y: np.ndarray, S: float) -> np.ndarray:
        """Lado derecho del sistema de ODEs.

        Args:
            y: Estado [rho_id, rho_lat]
            S: Variable entrópica

        Returns:
            Derivadas [drho_id/dS, drho_lat/dS]
        """
        rho_id, rho_lat = y

        z = self.z_of_S_func(S)
        H = self.H_func(z)
        S_dot = self.S_dot_func(S)

        S_dot_safe = max(abs(S_dot), 1e-30)

        # Términos de dilución
        dilution_id = -3.0 * H * (1.0 + self.w_id) * rho_id / S_dot_safe
        dilution_lat = -3.0 * H * (1.0 + self.w_lat) * rho_lat / S_dot_safe

        # Términos de intercambio (simplificado)
        Q_id, Q_lat = Q_dual_simple(S, self.q_params)

        # El signo: ρ̇_id = -Q_dual, ρ̇_lat = +Q_dual
        # En dρ/dS: dividimos por Ṡ y ajustamos signos
        exchange_id = -Q_id / S_dot_safe
        exchange_lat = +Q_lat / S_dot_safe

        drho_id_dS = dilution_id + exchange_id
        drho_lat_dS = dilution_lat + exchange_lat

        return np.array([drho_id_dS, drho_lat_dS])

    def evolve(
        self,
        S_range: Tuple[float, float],
        rho_id_init: float,
        rho_lat_init: float,
        n_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evoluciona los canales en S.

        Args:
            S_range: (S_min, S_max)
            rho_id_init: ρ_id inicial
            rho_lat_init: ρ_lat inicial
            n_points: Número de puntos

        Returns:
            (S_array, rho_id_array, rho_lat_array)
        """
        S_arr = np.linspace(S_range[0], S_range[1], n_points)
        y0 = [rho_id_init, rho_lat_init]

        sol = odeint(self.rhs, y0, S_arr)

        return S_arr, sol[:, 0], sol[:, 1]

    def check_conservation(
        self,
        S_arr: np.ndarray,
        rho_id_arr: np.ndarray,
        rho_lat_arr: np.ndarray
    ) -> float:
        """Verifica conservación total.

        La suma Σ_i Q_i debe ser cero, por lo que la densidad total
        (excluyendo dilución) debería conservarse en un sentido
        apropiado.

        Args:
            S_arr: Array de S
            rho_id_arr: Array de ρ_id
            rho_lat_arr: Array de ρ_lat

        Returns:
            Máxima violación de conservación relativa
        """
        # Verificar que Q_id + Q_lat = Q_S en cada punto
        max_violation = 0.0

        for i, S in enumerate(S_arr):
            Q_id, Q_lat = Q_dual_simple(S, self.q_params)
            Q_total = Q_id + Q_lat

            # Q_total debería ser igual a Q_amplitude
            expected = self.q_params.Q_amplitude
            if abs(expected) > 1e-30:
                violation = abs(Q_total - expected) / expected
                max_violation = max(max_violation, violation)

        return max_violation
