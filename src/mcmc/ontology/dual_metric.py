"""Métrica Dual Relativa g_μν(S) del MCMC.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001) - No existe geometría clásica
- Post-Big Bang: S ∈ [1.001, 95.07] - Cosmología observable

La Métrica Dual Relativa (MDR) es la cristalización geométrica de la
dualidad Mp/Ep. Incorpora la tensión pre-geométrica a través del
lapse entrópico N(S) = exp[Φ_ten(S)].

Métrica de fondo:
    ds² = -N²(S) dt²_rel + a²(S) δᵢⱼ dxⁱdxʲ

Componentes:
    g_tt(S) = -N²(S) = -exp[2Φ_ten(S)]
    g_ij(S) = a²(S) δ_ij

Límite ΛCDM: Φ_ten → 0 ⟹ N → 1 ⟹ métrica FRW estándar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .s_map import EntropyMap
from .adrian_field import AdrianField, AdrianFieldParams


@dataclass
class MDRParams:
    """Parámetros de la Métrica Dual Relativa.

    Attributes:
        G_eff: Constante gravitatoria efectiva
        k_curvature: Curvatura espacial (k=0 plano, +1 cerrado, -1 abierto)
        c: Velocidad de la luz (unidades naturales = 1)
    """
    G_eff: float = 1.0
    k_curvature: float = 0.0
    c: float = 1.0


class DualRelativeMetric:
    """Métrica Dual Relativa g_μν(S) del MCMC.

    Implementa la métrica que incorpora la tensión pre-geométrica
    del Campo de Adrián.

    Attributes:
        params: Parámetros de la MDR
        adrian_field: Campo de Adrián para Φ_ten
        entropy_map: Mapa entrópico para S↔z↔a
    """

    def __init__(
        self,
        params: MDRParams | None = None,
        adrian_field: AdrianField | None = None,
        entropy_map: EntropyMap | None = None
    ):
        """Inicializa la Métrica Dual Relativa.

        Args:
            params: Parámetros MDR
            adrian_field: Campo de Adrián
            entropy_map: Mapa entrópico
        """
        self.params = params or MDRParams()
        self.Phi_Ad = adrian_field or AdrianField()
        self.S_map = entropy_map or EntropyMap()

    # -------------------------------------------------------------------------
    # Componentes de la métrica
    # -------------------------------------------------------------------------

    def N_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Función lapse N(S) = exp[Φ_ten(S)].

        El lapse modula la componente temporal de la métrica.

        Args:
            S: Variable entrópica

        Returns:
            Función lapse N(S)
        """
        Phi_ten = self.Phi_Ad.Phi_ten(S)
        return np.exp(Phi_ten)

    def g_tt(self, S: float | np.ndarray) -> float | np.ndarray:
        """Componente temporal g_tt = -N²(S).

        Args:
            S: Variable entrópica

        Returns:
            Componente g_tt (negativa)
        """
        N = self.N_of_S(S)
        return -N**2

    def g_rr(self, S: float | np.ndarray) -> float | np.ndarray:
        """Componente radial g_rr = a²(S) / (1 - k r²).

        Simplificación: asumimos r << 1/sqrt|k| o k=0.

        Args:
            S: Variable entrópica

        Returns:
            Componente g_rr
        """
        a = self.a_of_S(S)
        return a**2

    def g_ij(self, S: float | np.ndarray, i: int, j: int) -> float | np.ndarray:
        """Componentes espaciales g_ij = a²(S) δ_ij.

        Args:
            S: Variable entrópica
            i, j: Índices espaciales (1, 2, 3)

        Returns:
            Componente g_ij
        """
        if i != j:
            return 0.0 if isinstance(S, (int, float)) else np.zeros_like(S)

        a = self.a_of_S(S)
        return a**2

    def a_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Factor de escala a(S).

        Args:
            S: Variable entrópica

        Returns:
            Factor de escala
        """
        return self.S_map.a_of_S(S)

    def sqrt_neg_g(self, S: float | np.ndarray) -> float | np.ndarray:
        """√(-g) = N · a³ para la métrica MDR.

        Determinante: g = g_tt · (g_rr)³ = -N² · a⁶

        Args:
            S: Variable entrópica

        Returns:
            √(-g)
        """
        N = self.N_of_S(S)
        a = self.a_of_S(S)
        return N * a**3

    # -------------------------------------------------------------------------
    # Derivadas y conexión
    # -------------------------------------------------------------------------

    def dN_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """Derivada del lapse dN/dS.

        dN/dS = N · dΦ_ten/dS

        Args:
            S: Variable entrópica

        Returns:
            Derivada dN/dS
        """
        N = self.N_of_S(S)
        dPhi_dS = self.Phi_Ad.dPhi_ten_dS(S)
        return N * dPhi_dS

    def da_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """Derivada del factor de escala da/dS.

        da/dS = a · (d ln a / dS) = a · C(S)

        Args:
            S: Variable entrópica

        Returns:
            Derivada da/dS
        """
        a = self.a_of_S(S)
        C = self.S_map.C_of_S(S)
        return a * C

    def H_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Parámetro de Hubble H(S) = (1/a)(da/dt).

        H = (1/a)(da/dS)(dS/dt) = C(S) / [T(S)·N(S)]

        Para conectar con observaciones, usamos H(z).

        Args:
            S: Variable entrópica

        Returns:
            Parámetro de Hubble H(S)
        """
        C = self.S_map.C_of_S(S)
        T = self.S_map.T_of_S(S)
        N = self.N_of_S(S)

        # Evitar división por cero
        denom = np.maximum(T * N, 1e-30)
        return C / denom

    # -------------------------------------------------------------------------
    # Curvatura
    # -------------------------------------------------------------------------

    def Ricci_scalar_FRW(self, H: float, dH_dt: float, a: float) -> float:
        """Escalar de Ricci R para FRW modificada.

        R = 6 [(ä/a) + (ȧ/a)² + k/a²]
          = 6 [dH/dt + 2H² + k/a²]

        Args:
            H: Parámetro de Hubble
            dH_dt: Derivada dH/dt
            a: Factor de escala

        Returns:
            Escalar de Ricci
        """
        k = self.params.k_curvature
        return 6.0 * (dH_dt + 2.0 * H**2 + k / a**2)

    def Ricci_scalar_S(
        self,
        S: float,
        H_of_z: Callable[[float], float] | None = None
    ) -> float:
        """Escalar de Ricci en función de S.

        Args:
            S: Variable entrópica
            H_of_z: Función H(z) para cálculo preciso (opcional)

        Returns:
            Escalar de Ricci
        """
        z = float(self.S_map.z_of_S(S))
        a = 1.0 / (1.0 + z)

        if H_of_z is not None:
            H = H_of_z(z)
            # Derivada numérica dH/dz
            dz = 1e-4
            dH_dz = (H_of_z(z + dz) - H_of_z(z - dz)) / (2 * dz)
            # dH/dt = dH/dz · dz/dt = -dH/dz · H · (1+z)
            dH_dt = -dH_dz * H * (1 + z)
        else:
            # Aproximación: H² ≈ constante
            H = float(self.H_of_S(S))
            dH_dt = 0.0

        return self.Ricci_scalar_FRW(H, dH_dt, a)

    # -------------------------------------------------------------------------
    # Límites y verificación
    # -------------------------------------------------------------------------

    def is_LCDM_limit(self, S: float | np.ndarray, tol: float = 1e-6) -> bool:
        """Verifica si estamos en el límite ΛCDM.

        En el límite ΛCDM: Φ_ten → 0 ⟹ N → 1 ⟹ métrica FRW estándar.

        Args:
            S: Variable entrópica
            tol: Tolerancia

        Returns:
            True si en límite ΛCDM
        """
        Phi_ten = self.Phi_Ad.Phi_ten(S)
        Phi_arr = np.atleast_1d(Phi_ten)
        return bool(np.all(np.abs(Phi_arr) < tol))

    def deviation_from_FRW(self, S: float | np.ndarray) -> float | np.ndarray:
        """Desviación de la métrica FRW.

        δg_tt / g_tt^{FRW} = N² - 1 = exp(2Φ_ten) - 1

        Args:
            S: Variable entrópica

        Returns:
            Desviación relativa
        """
        N = self.N_of_S(S)
        return N**2 - 1.0

    def compute_metric_tensor(self, S: float) -> np.ndarray:
        """Calcula el tensor métrico completo g_μν(S).

        Retorna matriz 4×4 en coordenadas (t, r, θ, φ) para métrica esférica.

        Args:
            S: Variable entrópica

        Returns:
            Matriz 4×4 del tensor métrico
        """
        g = np.zeros((4, 4))

        g_tt = float(self.g_tt(S))
        a2 = float(self.a_of_S(S))**2

        g[0, 0] = g_tt    # g_tt = -N²
        g[1, 1] = a2      # g_rr = a²
        g[2, 2] = a2      # g_θθ = a² (en coord. cartesianas)
        g[3, 3] = a2      # g_φφ = a²

        return g


# =============================================================================
# Funciones de conveniencia
# =============================================================================

def create_LCDM_metric() -> DualRelativeMetric:
    """Crea métrica en límite ΛCDM (Φ_ten = 0).

    Returns:
        MDR en límite ΛCDM
    """
    # Campo de Adrián con Φ_ten = 0
    params = AdrianFieldParams(phi_ten_amplitude=0.0)
    adrian = AdrianField(params)
    return DualRelativeMetric(adrian_field=adrian)


def create_MCMC_metric(
    phi_ten_amplitude: float = 0.01,
    phi_ten_center: float = 48.0,
    phi_ten_width: float = 10.0
) -> DualRelativeMetric:
    """Crea métrica MDR con desviación de ΛCDM.

    CORRECCIÓN: S ∈ [0, 100], post-Big Bang S ∈ [1.001, 95.07]

    Args:
        phi_ten_amplitude: Amplitud de Φ_ten
        phi_ten_center: Centro de la transición (default: ~48, mitad post-BB)
        phi_ten_width: Anchura de la transición (default: 10 para rango amplio)

    Returns:
        MDR con tensión no nula
    """
    params = AdrianFieldParams(
        phi_ten_amplitude=phi_ten_amplitude,
        phi_ten_center=phi_ten_center,
        phi_ten_width=phi_ten_width
    )
    adrian = AdrianField(params)
    return DualRelativeMetric(adrian_field=adrian)
