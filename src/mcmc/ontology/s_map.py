"""Mapa Entrópico Completo S↔z↔t↔a.

CORRECCIÓN ONTOLÓGICA (2025):
El parámetro entrópico S tiene rango [0, 100], NO [1.001, 1.0015].

RÉGIMEN PRE-GEOMÉTRICO: S ∈ [0, 1.001)
- No existe espacio-tiempo clásico
- Transiciones canónicas preservadas

RÉGIMEN GEOMÉTRICO (POST-BIG BANG): S ∈ [1.001, 95.07]
Ecuación maestra reformulada:
    S(z) = S_geom + (S_0 - S_geom) / E(z)²

donde E(z) = H(z)/H_0 = √[Ω_m(1+z)³ + Ω_Λ]

Esta formulación garantiza:
    - Monotonía: dS/dz < 0 (mayor z implica menor S)
    - Límites: S(0) = S_0 ≈ 95, S(∞) → S_geom = 1.001 (Big Bang)
    - Consistencia termodinámica: S_H ∝ 1/H² (Bekenstein-Hawking)

Presente estratificado:
    S_local(x) = S_global × √(1 - 2GM/rc²)

Las islas tensoriales (BH, cúmulos) experimentan S_local < S_global.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import quad

from mcmc.core.ontology import (
    OMEGA_M, OMEGA_LAMBDA,
    S_GEOM, S_0, G, c, RHO_CRIT
)


@dataclass
class SMapParams:
    """Parámetros del mapa S↔z↔t↔a.

    CORRECCIÓN: Rango S ∈ [0, 100]
    - Pre-geométrico: S ∈ [0, 1.001)
    - Post-Big Bang: S ∈ [1.001, 95.07]

    Attributes:
        S_GEOM: Big Bang - transición pre-geométrica (= 1.001)
        S_0: S actual (hoy, z=0) ≈ 95.07
        Omega_m: Fracción de materia total
        Omega_Lambda: Fracción de energía oscura
    """
    S_GEOM: float = S_GEOM
    S_0: float = S_0
    Omega_m: float = OMEGA_M
    Omega_Lambda: float = OMEGA_LAMBDA


class EntropyMap:
    """Mapa completo S↔z↔t↔a con presente estratificado.

    CORRECCIÓN ONTOLÓGICA: S ∈ [0, 100]

    Proporciona conversiones bidireccionales entre:
    - Variable entrópica S
    - Redshift z
    - Tiempo cósmico t
    - Factor de escala a

    Attributes:
        params: Parámetros del mapa
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

    # =========================================================================
    # Parámetro de Hubble normalizado
    # =========================================================================

    def E_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """E(z) = H(z)/H_0 = √[Ω_m(1+z)³ + Ω_Λ].

        Args:
            z: Redshift

        Returns:
            Parámetro de Hubble normalizado
        """
        z_arr = np.asarray(z)
        p = self.params
        return np.sqrt(p.Omega_m * (1 + z_arr)**3 + p.Omega_Lambda)

    def E_squared(self, z: float | np.ndarray) -> float | np.ndarray:
        """E²(z) = Ω_m(1+z)³ + Ω_Λ.

        Args:
            z: Redshift

        Returns:
            E² para cálculos de S
        """
        z_arr = np.asarray(z)
        p = self.params
        return p.Omega_m * (1 + z_arr)**3 + p.Omega_Lambda

    # =========================================================================
    # Mapeo S ↔ z (CORREGIDO)
    # =========================================================================

    def S_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """S(z) según la ecuación maestra reformulada.

        ECUACIÓN CORREGIDA (solo válida para régimen post-Big Bang):
        S(z) = S_geom + (S_0 - S_geom) / E(z)²

        Basada en entropía del horizonte de Hubble: S_H ∝ 1/H²

        Propiedades:
        - S(z=0) = S_0 ≈ 95.07 (hoy)
        - S(z→∞) → S_geom = 1.001 (Big Bang)
        - dS/dz < 0 (monótona decreciente)

        Args:
            z: Redshift (z ≥ 0)

        Returns:
            Variable entrópica S ∈ [S_geom, S_0]
        """
        z_arr = np.asarray(z)
        if np.any(z_arr < 0):
            raise ValueError("z debe ser ≥ 0")

        p = self.params
        E2 = self.E_squared(z_arr)
        delta_S = p.S_0 - p.S_GEOM

        S = p.S_GEOM + delta_S / E2
        return float(S) if np.ndim(S) == 0 else S

    def z_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """z(S) - inversión del mapa S(z).

        INVERSA:
        z = [(S_0 - S_geom)/(S - S_geom)/Ω_m - Ω_Λ/Ω_m]^(1/3) - 1

        Args:
            S: Variable entrópica (S_geom < S ≤ S_0)

        Returns:
            Redshift z ≥ 0
        """
        S_arr = np.asarray(S)
        p = self.params

        if np.any(S_arr <= p.S_GEOM) or np.any(S_arr > p.S_0 + 0.1):
            raise ValueError(f"S debe estar en ({p.S_GEOM}, {p.S_0}]")

        delta_S = p.S_0 - p.S_GEOM
        E2 = delta_S / (S_arr - p.S_GEOM)

        # E² = Ω_m(1+z)³ + Ω_Λ
        # (1+z)³ = (E² - Ω_Λ) / Ω_m
        one_plus_z_cubed = (E2 - p.Omega_Lambda) / p.Omega_m
        one_plus_z_cubed = np.maximum(one_plus_z_cubed, 1.0)  # z ≥ 0

        z = one_plus_z_cubed ** (1/3) - 1
        return float(z) if np.ndim(z) == 0 else z

    def dS_dz(self, z: float | np.ndarray) -> float | np.ndarray:
        """dS/dz - derivada del mapa S(z).

        dS/dz = -(S_0 - S_geom) × d(1/E²)/dz
              = -(S_0 - S_geom) × (-2/E³) × dE/dz

        Args:
            z: Redshift

        Returns:
            Derivada dS/dz (negativa, S decrece con z)
        """
        z_arr = np.asarray(z)
        p = self.params

        E = self.E_of_z(z_arr)
        E2 = E**2
        delta_S = p.S_0 - p.S_GEOM

        # dE²/dz = 3 Ω_m (1+z)²
        dE2_dz = 3 * p.Omega_m * (1 + z_arr)**2

        # d(1/E²)/dz = -dE²/dz / E⁴
        dS_dz = -delta_S * dE2_dz / E2**2

        return dS_dz

    # =========================================================================
    # Mapeo S ↔ a
    # =========================================================================

    def a_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Factor de escala a(S) = 1/(1+z(S)).

        Args:
            S: Variable entrópica

        Returns:
            Factor de escala a ∈ (0, 1]
        """
        z = self.z_of_S(S)
        return 1.0 / (1.0 + z)

    def S_of_a(self, a: float | np.ndarray) -> float | np.ndarray:
        """S(a) vía z = 1/a - 1.

        Args:
            a: Factor de escala (0 < a ≤ 1)

        Returns:
            Variable entrópica S
        """
        a_arr = np.asarray(a)
        z = 1.0 / a_arr - 1.0
        return self.S_of_z(z)

    # =========================================================================
    # Funciones de la Ley de Cronos
    # =========================================================================

    def T_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """T(S) - función de cronificación base.

        En el nuevo esquema, T(S) ∝ S para épocas tardías.

        Args:
            S: Variable entrópica

        Returns:
            Función de cronificación
        """
        S_arr = np.asarray(S)
        p = self.params

        # T normalizado para dar edad correcta del universo
        T_0 = 13.8e9  # años (edad del universo)
        return T_0 * S_arr / p.S_0

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

        En el límite ΛCDM: Φ_ten → 0 ⟹ N → 1

        Args:
            S: Variable entrópica

        Returns:
            Lapse entrópico
        """
        return np.exp(self.phi_ten(S))

    def dt_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """dt/dS según Ley de Cronos reformulada.

        dt/dS ∝ T(S) × N(S) / |dS/dz| × |dz/dt|

        Args:
            S: Variable entrópica

        Returns:
            Derivada dt/dS
        """
        # Simplificación: en épocas tardías dt/dS ≈ constante × N(S)
        N = self.N_of_S(S)
        # Normalizar para edad correcta
        t_0 = 13.8e9  # años
        p = self.params
        return t_0 / p.S_0 * N

    # =========================================================================
    # Mapeo S ↔ t (tiempo cósmico)
    # =========================================================================

    def t_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Tiempo cósmico t(S) en años.

        Integración numérica de dt/dS desde S_GEOM.

        Args:
            S: Variable entrópica

        Returns:
            Tiempo cósmico en años
        """
        S_arr = np.atleast_1d(S)
        results = np.zeros_like(S_arr, dtype=float)
        p = self.params

        for i, s in enumerate(S_arr):
            if s <= p.S_GEOM:
                results[i] = 0.0
            else:
                integral, _ = quad(lambda x: self.dt_dS(x), p.S_GEOM, s)
                results[i] = integral

        return results[0] if np.ndim(S) == 0 else results

    def t_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """Tiempo cósmico t(z) en años.

        Args:
            z: Redshift

        Returns:
            Tiempo cósmico en años
        """
        S = self.S_of_z(z)
        return self.t_of_S(S)

    # =========================================================================
    # Presente estratificado
    # =========================================================================

    def S_local(
        self,
        r: float,
        M: float,
        S_glob: float | None = None
    ) -> float:
        """S_local en una isla tensorial.

        PRESENTE ESTRATIFICADO:
        S_local = S_global × √(1 - 2GM/rc²)

        Las regiones de alta densidad gravitacional tienen S_local < S_global.

        Args:
            r: Distancia al centro de la isla tensorial [m]
            M: Masa de la isla tensorial [kg]
            S_glob: S global (default: S_0)

        Returns:
            S_local (siempre ≤ S_global)
        """
        if S_glob is None:
            S_glob = self.params.S_0

        r_s = 2 * G * M / c**2  # Radio de Schwarzschild
        if r <= r_s:
            return 0.0  # En el horizonte, S_local → 0

        xi = G * M / (r * c**2)
        f_dilatacion = np.sqrt(1 - 2 * xi)
        return S_glob * f_dilatacion

    def S_global(
        self,
        z: float = 0,
        exclude_islands: bool = True
    ) -> float:
        """S_global promediado fuera de islas tensoriales.

        Args:
            z: Redshift para calcular S base
            exclude_islands: Si True, aplica corrección por islas

        Returns:
            S_global del fondo cosmológico
        """
        S_base = self.S_of_z(z)

        if exclude_islands:
            # ~2% de masa en estructuras densas
            f_cluster = 0.02
            delta_S_rel = 0.1
            correction = 1 / (1 - f_cluster * delta_S_rel)
            S_g = float(S_base) * correction
        else:
            S_g = float(S_base)

        return min(S_g, self.params.S_0)

    def tiempo_atraso_isla(
        self,
        M: float,
        r: float,
        t_cosmico: float | None = None
    ) -> float:
        """Atraso temporal de una isla tensorial.

        Δτ_atraso ≈ t_cósmico × (GM/rc²)

        Args:
            M: Masa de la isla [kg]
            r: Distancia al centro [m]
            t_cosmico: Tiempo de referencia [años] (default: edad del universo)

        Returns:
            Atraso temporal [años]
        """
        if t_cosmico is None:
            t_cosmico = 13.8e9  # años

        xi = G * M / (r * c**2)
        return t_cosmico * xi

    # =========================================================================
    # Densidades ρ_id y ρ_lat
    # =========================================================================

    def rho_id(self, S: float, z: float = 0) -> float:
        """Densidad ideal (masa determinada) en función de S.

        ρ_id = ρ_crit × (100 - S)/100 × Ω_m × (1+z)³

        Args:
            S: Variable entrópica
            z: Redshift

        Returns:
            Densidad ideal [kg/m³]
        """
        p = self.params
        f_determinada = (p.S_0 - S) / p.S_0  # Fracción no convertida
        f_determinada = max(0, f_determinada)
        return RHO_CRIT * f_determinada * p.Omega_m * (1 + z)**3

    def rho_lat(self, S: float, z: float = 0) -> float:
        """Densidad latente (espacio potencial Ep) en función de S.

        ρ_lat = ρ_crit × (S/S_0) × Ω_Λ

        Args:
            S: Variable entrópica
            z: Redshift (no afecta Λ en ΛCDM)

        Returns:
            Densidad latente [kg/m³]
        """
        p = self.params
        f_espacial = S / p.S_0  # Fracción convertida
        f_espacial = min(1, max(0, f_espacial))
        return RHO_CRIT * f_espacial * p.Omega_Lambda

    def Q_dual(self, S: float, dS_dt: float) -> float:
        """Término de transferencia dual.

        Q_dual = -(dS/dt) × (ρ_id - ρ_lat) / S

        Captura la transferencia entre masa determinada y espacio.

        Args:
            S: Variable entrópica
            dS_dt: Derivada temporal de S

        Returns:
            Término de intercambio Q_dual
        """
        if S <= 0:
            return 0.0

        delta_rho = self.rho_id(S) - self.rho_lat(S)
        return -dS_dt * delta_rho / S

    # =========================================================================
    # Verificaciones
    # =========================================================================

    def is_LCDM_limit(self, tol: float = 1e-6) -> bool:
        """Verifica si estamos en el límite ΛCDM (Φ_ten ≈ 0).

        Args:
            tol: Tolerancia

        Returns:
            True si Φ_ten es despreciable
        """
        S_test = np.linspace(self.params.S_GEOM, self.params.S_0, 10)
        phi_vals = self.phi_ten(S_test)
        return bool(np.all(np.abs(phi_vals) < tol))

    def verify_monotonicity(self) -> tuple[bool, str]:
        """Verifica monotonicidad de los mapeos.

        Returns:
            (is_valid, message)
        """
        z_test = np.linspace(0, 1000, 100)
        S_test = self.S_of_z(z_test)

        # dS/dz debe ser negativo (S decrece con z)
        dS = np.diff(S_test)
        dz = np.diff(z_test)
        dS_dz_numerical = dS / dz

        if not np.all(dS_dz_numerical < 0):
            return False, "S(z) no es monótona decreciente"

        return True, "OK: S(z) es monótona decreciente"


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


def S_of_z_simple(
    z: np.ndarray,
    S_GEOM: float = S_GEOM,
    S_0: float = S_0,
    Omega_m: float = OMEGA_M,
    Omega_Lambda: float = OMEGA_LAMBDA
) -> np.ndarray:
    """Versión simplificada de S(z) sin crear objeto.

    S(z) = S_geom + (S_0 - S_geom) / E(z)²

    Args:
        z: Redshift array
        S_GEOM: Transición pre-geométrica
        S_0: S actual
        Omega_m: Fracción de materia
        Omega_Lambda: Fracción de energía oscura

    Returns:
        S array
    """
    z = np.asarray(z)
    E2 = Omega_m * (1 + z)**3 + Omega_Lambda
    return S_GEOM + (S_0 - S_GEOM) / E2


def z_of_S_simple(
    S: np.ndarray,
    S_GEOM: float = S_GEOM,
    S_0: float = S_0,
    Omega_m: float = OMEGA_M,
    Omega_Lambda: float = OMEGA_LAMBDA
) -> np.ndarray:
    """Versión simplificada de z(S) sin crear objeto.

    z = [(S_0 - S_geom)/(S - S_geom)/Ω_m - Ω_Λ/Ω_m]^(1/3) - 1

    Args:
        S: Entropic variable array
        S_GEOM: Transición pre-geométrica
        S_0: S actual
        Omega_m: Fracción de materia
        Omega_Lambda: Fracción de energía oscura

    Returns:
        z array
    """
    S = np.asarray(S)
    E2 = (S_0 - S_GEOM) / np.maximum(S - S_GEOM, 1e-10)
    one_plus_z_cubed = np.maximum((E2 - Omega_Lambda) / Omega_m, 1.0)
    return one_plus_z_cubed ** (1/3) - 1


# Alias para compatibilidad
def ley_de_cronos(S, s_map):
    """Alias: dt_rel/dS."""
    return s_map.dt_dS(S)


def t_rel_of_S(S, s_map):
    """Alias: t(S)."""
    return s_map.t_of_S(S)


S_of_z_post_BB = S_of_z_simple
z_of_S_post_BB = z_of_S_simple
