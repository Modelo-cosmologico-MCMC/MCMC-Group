"""Campo de Adrián Φ_Ad - Regulador tensional fundamental.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]

RÉGIMEN PRE-GEOMÉTRICO: S ∈ [0, 1.001)
- Transiciones canónicas preservadas (S = 0.001, 0.01, 0.1, 0.5)
- No existe espacio-tiempo clásico
- Φ_Ad modula la tensión Mp/Ep primordial

RÉGIMEN GEOMÉTRICO (POST-BIG BANG): S ∈ [1.001, 95.07]
- Big Bang en S = 1.001
- Transiciones post-Big Bang (formación estructuras, presente)

El Campo de Adrián media la tensión Mp/Ep mediante dos componentes:
    - Faz escalar Φ_esc: Liberación de energía en transiciones
    - Faz tensorial Φ_ten: Estructura relacional y curvatura

Potencial tensional canónico:
    V(Φ_Ad; S) = V₀ + α_S·S·Φ²_Ad + Σ_n β_n(Φ²_Ad - v²_n)²·Θ_λ(S - S_n)

donde Θ_λ(x) = ½[1 + tanh(x/λ)] es un escalón suavizado.

Ecuación de movimiento:
    K(S) ∂²_S Φ_Ad + (dK/dS) ∂_S Φ_Ad + δV/δΦ_Ad = 0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.integrate import odeint

from mcmc.core.ontology import S_0


@dataclass
class AdrianFieldParams:
    """Parámetros del Campo de Adrián.

    CORRECCIÓN: S ∈ [0, 100]
    - Pre-geométrico: S ∈ [0, 1.001)
    - Post-Big Bang: S ∈ [1.001, 95.07]

    Attributes:
        V0: Nivel base del vacío (energía mínima)
        alpha_S: Acoplamiento S-Φ² (masa efectiva dependiente de S)
        lambda_4: Autointeracción cuártica
        v_vac: VEV (valor esperado del vacío, escala electrodébil)
        K_0: Rigidez tensorial base K(S)
        K_slope: Variación de K con S
        lambda_smooth: Anchura de suavización de Θ_λ
    """
    V0: float = 1.0
    alpha_S: float = 0.0001  # Reducido para nuevo rango de S
    lambda_4: float = 0.1
    v_vac: float = 246.0  # GeV (escala Higgs)
    K_0: float = 1.0
    K_slope: float = 0.0
    lambda_smooth: float = 0.1  # Menor para transiciones más precisas

    # Parámetros para Φ_ten
    # CORRECCIÓN: phi_ten_center ahora en régimen post-Big Bang
    phi_ten_amplitude: float = 0.01
    phi_ten_center: float = 48.0   # Mitad del rango post-Big Bang
    phi_ten_width: float = 10.0    # Mayor para S ∈ [1.001, 95]


@dataclass
class TransitionParams:
    """Parámetros de una transición ontológica.

    Attributes:
        S_n: Umbral de la transición
        v_n: VEV en esta fase
        beta_n: Coeficiente de la barrera
    """
    S_n: float
    v_n: float
    beta_n: float = 1.0


class AdrianField:
    """Campo de Adrián con faz escalar y tensorial.

    CORRECCIÓN ONTOLÓGICA: S ∈ [0, 100]
    - Pre-geométrico: S ∈ [0, 1.001) con transiciones canónicas
    - Post-Big Bang: S ∈ [1.001, 95.07]

    Implementa el regulador tensorial fundamental del MCMC que media
    la dualidad Masa Primordial / Espacio Primordial.

    Attributes:
        params: Parámetros del campo
        transitions: Lista de transiciones ontológicas (pre y post Big Bang)
    """

    def __init__(
        self,
        params: AdrianFieldParams | None = None,
        transitions: List[TransitionParams] | None = None
    ):
        """Inicializa el Campo de Adrián.

        Args:
            params: Parámetros del campo
            transitions: Lista de transiciones (defaults a umbrales canónicos)
        """
        self.params = params or AdrianFieldParams()

        if transitions is None:
            # Transiciones canónicas preservando régimen pre-geométrico
            # PRE-GEOMÉTRICO: S ∈ [0, 1.001)
            # POST-BIG BANG: S ∈ [1.001, 95.07]
            self.transitions = [
                # === Régimen Pre-Geométrico (S < 1.001) ===
                TransitionParams(S_n=0.001, v_n=0.0, beta_n=0.05),  # Primordial
                TransitionParams(S_n=0.01, v_n=0.0, beta_n=0.1),    # Trans pre-geom 1
                TransitionParams(S_n=0.1, v_n=0.0, beta_n=0.2),     # Trans pre-geom 2
                TransitionParams(S_n=0.5, v_n=0.0, beta_n=0.3),     # Trans pre-geom 3
                # === Big Bang: S = 1.001 ===
                TransitionParams(S_n=1.001, v_n=0.0, beta_n=1.0),   # Big Bang
                # === Régimen Post-Big Bang (S ≥ 1.001) ===
                TransitionParams(S_n=2.0, v_n=0.0, beta_n=0.8),     # Primeras estructuras
                TransitionParams(S_n=47.5, v_n=246.0, beta_n=1.0),  # Pico formación estelar
                TransitionParams(S_n=95.0, v_n=246.0, beta_n=0.1),  # Presente
            ]
        else:
            self.transitions = transitions

    # -------------------------------------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------------------------------------

    def Theta_lambda(self, x: float | np.ndarray) -> float | np.ndarray:
        """Escalón suave Θ_λ(x) = ½[1 + tanh(x/λ)].

        Args:
            x: Argumento

        Returns:
            Valor del escalón suavizado (0 a 1)
        """
        lam = max(self.params.lambda_smooth, 1e-12)
        return 0.5 * (1.0 + np.tanh(np.asarray(x) / lam))

    def W_n(self, S: float, n: int) -> float:
        """Función ventana para el tramo n.

        W_n(S) = Θ_λ(S - S_n) - Θ_λ(S - S_{n+1})

        Selecciona el intervalo [S_n, S_{n+1}).

        Args:
            S: Variable entrópica
            n: Índice del tramo

        Returns:
            Peso de la ventana (0 a 1)
        """
        if n >= len(self.transitions):
            return 0.0

        S_n = self.transitions[n].S_n
        S_np1 = self.transitions[n + 1].S_n if n + 1 < len(self.transitions) else np.inf

        return float(self.Theta_lambda(S - S_n) - self.Theta_lambda(S - S_np1))

    def K_of_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Rigidez tensional K(S).

        K(S) = K_0 + K_slope · S

        Args:
            S: Variable entrópica

        Returns:
            Rigidez tensorial
        """
        return self.params.K_0 + self.params.K_slope * np.asarray(S)

    def dK_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """Derivada de la rigidez dK/dS.

        Args:
            S: Variable entrópica

        Returns:
            Derivada dK/dS
        """
        return np.full_like(np.asarray(S), self.params.K_slope, dtype=float)

    # -------------------------------------------------------------------------
    # Potencial V(Φ; S)
    # -------------------------------------------------------------------------

    def V_base(self, Phi: float | np.ndarray, S: float) -> float | np.ndarray:
        """Parte base del potencial.

        V_base = V₀ + α_S · S · Φ²

        Args:
            Phi: Campo Φ_Ad
            S: Variable entrópica

        Returns:
            Potencial base
        """
        p = self.params
        Phi_arr = np.asarray(Phi)
        return p.V0 + p.alpha_S * S * Phi_arr**2

    def V_transition(self, Phi: float | np.ndarray, S: float) -> float | np.ndarray:
        """Contribución de transiciones al potencial.

        V_trans = Σ_n β_n(Φ² - v²_n)² · Θ_λ(S - S_n)

        Args:
            Phi: Campo Φ_Ad
            S: Variable entrópica

        Returns:
            Potencial de transiciones
        """
        Phi_arr = np.asarray(Phi)
        V_trans = np.zeros_like(Phi_arr, dtype=float)

        for trans in self.transitions:
            theta = self.Theta_lambda(S - trans.S_n)
            V_trans += trans.beta_n * (Phi_arr**2 - trans.v_n**2)**2 * theta

        return V_trans

    def V_eff(self, Phi: float | np.ndarray, S: float) -> float | np.ndarray:
        """Potencial efectivo total V(Φ_Ad; S).

        V(Φ; S) = V₀ + α_S·S·Φ² + Σ_n β_n(Φ² - v²_n)²·Θ_λ(S - S_n)

        Args:
            Phi: Campo Φ_Ad
            S: Variable entrópica

        Returns:
            Potencial total
        """
        return self.V_base(Phi, S) + self.V_transition(Phi, S)

    def dV_dPhi(self, Phi: float | np.ndarray, S: float) -> float | np.ndarray:
        """Derivada del potencial respecto a Φ.

        δV/δΦ = 2α_S·S·Φ + Σ_n 4β_n·Φ·(Φ² - v²_n)·Θ_λ(S - S_n)

        Args:
            Phi: Campo Φ_Ad
            S: Variable entrópica

        Returns:
            Derivada δV/δΦ
        """
        p = self.params
        Phi_arr = np.asarray(Phi)

        # Término de masa
        dV = 2 * p.alpha_S * S * Phi_arr

        # Términos de transición
        for trans in self.transitions:
            theta = self.Theta_lambda(S - trans.S_n)
            dV += 4 * trans.beta_n * Phi_arr * (Phi_arr**2 - trans.v_n**2) * theta

        return dV

    def dV_dS(self, Phi: float | np.ndarray, S: float) -> float | np.ndarray:
        """Derivada del potencial respecto a S.

        Args:
            Phi: Campo Φ_Ad
            S: Variable entrópica

        Returns:
            Derivada ∂V/∂S
        """
        p = self.params
        Phi_arr = np.asarray(Phi)

        # d/dS de V_base
        dV = p.alpha_S * Phi_arr**2

        # d/dS de Θ_λ(S - S_n)
        lam = max(p.lambda_smooth, 1e-12)
        for trans in self.transitions:
            arg = (S - trans.S_n) / lam
            dTheta_dS = 0.5 * (1.0 - np.tanh(arg)**2) / lam
            dV += trans.beta_n * (Phi_arr**2 - trans.v_n**2)**2 * dTheta_dS

        return dV

    # -------------------------------------------------------------------------
    # Fazes del Campo
    # -------------------------------------------------------------------------

    def Phi_ten(self, S: float | np.ndarray) -> float | np.ndarray:
        """Faz tensorial Φ_ten(S) - modula g_tt.

        Parametrización simple: transición suave.

        Φ_ten(S) = A · tanh[(S - S_c) / w]

        El lapse es N(S) = exp[Φ_ten(S)]

        Args:
            S: Variable entrópica (ahora en [0, 100])

        Returns:
            Faz tensorial Φ_ten
        """
        p = self.params
        S_arr = np.asarray(S)
        arg = (S_arr - p.phi_ten_center) / max(p.phi_ten_width, 1e-12)
        return p.phi_ten_amplitude * np.tanh(arg)

    def dPhi_ten_dS(self, S: float | np.ndarray) -> float | np.ndarray:
        """Derivada de Φ_ten respecto a S.

        Args:
            S: Variable entrópica

        Returns:
            Derivada dΦ_ten/dS
        """
        p = self.params
        S_arr = np.asarray(S)
        w = max(p.phi_ten_width, 1e-12)
        arg = (S_arr - p.phi_ten_center) / w
        return (p.phi_ten_amplitude / w) * (1.0 - np.tanh(arg)**2)

    def Phi_esc(self, S: float | np.ndarray) -> float | np.ndarray:
        """Faz escalar Φ_esc(S) - balance Mp/Ep.

        Solución aproximada en el mínimo de V_eff:
        En el mínimo, dV/dΦ = 0.

        Aproximación simple: Φ_esc decrece con S hacia el VEV.

        Args:
            S: Variable entrópica (ahora en [0, 100])

        Returns:
            Faz escalar Φ_esc
        """
        p = self.params
        S_arr = np.asarray(S)

        # CORRECCIÓN: Usar S_0 ≈ 95 en lugar de 1.001
        S_ref = S_0  # ≈ 95.07
        transition = self.Theta_lambda(S_arr - S_ref * 0.5)

        # Interpolación simple hacia el VEV
        Phi_early = p.v_vac * np.sqrt(np.maximum(1 - (S_arr / S_ref)**2, 0))
        Phi_late = p.v_vac

        return (1 - transition) * Phi_early + transition * Phi_late

    # -------------------------------------------------------------------------
    # Ecuación de movimiento
    # -------------------------------------------------------------------------

    def eom_rhs(self, y: np.ndarray, S: float) -> np.ndarray:
        """Lado derecho de la ecuación de movimiento.

        K(S) Φ'' + K'(S) Φ' + dV/dΦ = 0

        ⟹ Φ'' = -[K'(S)/K(S)] Φ' - [1/K(S)] dV/dΦ

        Sistema: y = [Φ, Φ']

        Args:
            y: Estado [Φ, dΦ/dS]
            S: Variable entrópica

        Returns:
            Derivadas [dΦ/dS, d²Φ/dS²]
        """
        Phi, dPhi_dS = y

        K = self.K_of_S(S)
        dK = self.dK_dS(S)
        dV = self.dV_dPhi(Phi, S)

        K = max(K, 1e-12)

        d2Phi_dS2 = -(dK / K) * dPhi_dS - (1 / K) * dV

        return np.array([dPhi_dS, d2Phi_dS2])

    def solve_eom(
        self,
        S_range: tuple[float, float],
        Phi_init: float,
        dPhi_init: float,
        n_points: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resuelve la ecuación de movimiento.

        Args:
            S_range: (S_min, S_max)
            Phi_init: Valor inicial de Φ
            dPhi_init: Valor inicial de dΦ/dS
            n_points: Número de puntos

        Returns:
            (S_arr, Phi_arr, dPhi_arr)
        """
        S_arr = np.linspace(S_range[0], S_range[1], n_points)
        y0 = [Phi_init, dPhi_init]

        sol = odeint(self.eom_rhs, y0, S_arr)

        return S_arr, sol[:, 0], sol[:, 1]

    # -------------------------------------------------------------------------
    # Densidades y presiones
    # -------------------------------------------------------------------------

    def rho_Ad(
        self,
        Phi: float,
        dPhi_dS: float,
        S: float,
        dS_dt: float
    ) -> float:
        """Densidad de energía del Campo de Adrián.

        ρ_Ad = ½ K(S) (dΦ/dt)² + V(Φ; S)
             = ½ K(S) (dΦ/dS)² (dS/dt)² + V(Φ; S)

        Args:
            Phi: Campo
            dPhi_dS: Derivada dΦ/dS
            S: Variable entrópica
            dS_dt: Derivada dS/dt

        Returns:
            Densidad de energía
        """
        K = self.K_of_S(S)
        V = self.V_eff(Phi, S)
        kinetic = 0.5 * K * dPhi_dS**2 * dS_dt**2
        return float(kinetic + V)

    def p_Ad(
        self,
        Phi: float,
        dPhi_dS: float,
        S: float,
        dS_dt: float
    ) -> float:
        """Presión del Campo de Adrián.

        p_Ad = ½ K(S) (dΦ/dt)² - V(Φ; S)

        Args:
            Phi: Campo
            dPhi_dS: Derivada dΦ/dS
            S: Variable entrópica
            dS_dt: Derivada dS/dt

        Returns:
            Presión
        """
        K = self.K_of_S(S)
        V = self.V_eff(Phi, S)
        kinetic = 0.5 * K * dPhi_dS**2 * dS_dt**2
        return float(kinetic - V)

    def w_Ad(self, Phi: float, dPhi_dS: float, S: float, dS_dt: float) -> float:
        """Ecuación de estado w_Ad = p_Ad/ρ_Ad.

        Cuando Φ̇² << V: w_Ad ≈ -1 (comportamiento tipo DE)

        Args:
            Phi: Campo
            dPhi_dS: Derivada dΦ/dS
            S: Variable entrópica
            dS_dt: Derivada dS/dt

        Returns:
            Ecuación de estado w_Ad
        """
        rho = self.rho_Ad(Phi, dPhi_dS, S, dS_dt)
        p = self.p_Ad(Phi, dPhi_dS, S, dS_dt)

        if abs(rho) < 1e-30:
            return -1.0

        return p / rho
