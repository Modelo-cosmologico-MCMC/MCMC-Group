"""Parámetros de gravedad modificada μ(a) y η(a).

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001) - No hay gravedad clásica
- Post-Big Bang: S ∈ [1.001, 95.07] - Gravedad modificada parametrizada

En el MCMC, las desviaciones de la gravedad de GR se parametrizan
mediante dos funciones fenomenológicas:

    μ(a; S) = G_eff / G_N  (modificación de Newton)
    η(a; S) = Φ / Ψ        (slip gravitacional)

donde Φ y Ψ son los potenciales de Bardeen.

Para GR: μ = η = 1.

En el MCMC, estas desviaciones provienen de:
    1. El Campo de Adrián Φ_Ad que modifica la dinámica
    2. La tensión pre-geométrica del lapse N(S)
    3. Acoplos efectivos cerca de transiciones

Observaciones de RSD y lensing constrañen combinaciones:
    Σ = μ · (1 + η) / 2  (lensing)
    Υ = μ · f            (RSD)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np

from ..ontology.s_map import EntropyMap


@dataclass
class MuEtaParams:
    """Parámetros para μ(a) y η(a).

    Attributes:
        mu_0: Valor de μ a a=1 (hoy)
        mu_a: Pendiente dμ/da a a=1
        eta_0: Valor de η a a=1
        eta_a: Pendiente dη/da a a=1
        a_trans: Escala de transición (si aplica)
        delta_a: Anchura de transición
    """
    mu_0: float = 1.0
    mu_a: float = 0.0
    eta_0: float = 1.0
    eta_a: float = 0.0
    a_trans: float = 0.5
    delta_a: float = 0.1


# =============================================================================
# Parametrizaciones de μ(a)
# =============================================================================

def mu_CPL(a: float | np.ndarray, params: MuEtaParams) -> float | np.ndarray:
    """Parametrización CPL-like para μ(a).

    μ(a) = μ_0 + μ_a · (1 - a)

    Similar a w(a) = w_0 + w_a(1-a).

    Args:
        a: Factor de escala
        params: Parámetros

    Returns:
        μ(a)
    """
    a_arr = np.asarray(a)
    return params.mu_0 + params.mu_a * (1.0 - a_arr)


def eta_CPL(a: float | np.ndarray, params: MuEtaParams) -> float | np.ndarray:
    """Parametrización CPL-like para η(a).

    η(a) = η_0 + η_a · (1 - a)

    Args:
        a: Factor de escala
        params: Parámetros

    Returns:
        η(a)
    """
    a_arr = np.asarray(a)
    return params.eta_0 + params.eta_a * (1.0 - a_arr)


def mu_tanh_transition(
    a: float | np.ndarray,
    params: MuEtaParams,
    mu_early: float = 1.0,
    mu_late: float = 1.0
) -> float | np.ndarray:
    """μ(a) con transición suave tipo tanh.

    μ(a) = μ_early + (μ_late - μ_early) · ½[1 + tanh((a - a_trans)/δ_a)]

    Args:
        a: Factor de escala
        params: Parámetros
        mu_early: Valor en épocas tempranas (a << a_trans)
        mu_late: Valor en épocas tardías (a >> a_trans)

    Returns:
        μ(a)
    """
    a_arr = np.asarray(a)
    x = (a_arr - params.a_trans) / max(params.delta_a, 1e-6)
    transition = 0.5 * (1.0 + np.tanh(x))
    return mu_early + (mu_late - mu_early) * transition


def eta_tanh_transition(
    a: float | np.ndarray,
    params: MuEtaParams,
    eta_early: float = 1.0,
    eta_late: float = 1.0
) -> float | np.ndarray:
    """η(a) con transición suave tipo tanh.

    Args:
        a: Factor de escala
        params: Parámetros
        eta_early: Valor en épocas tempranas
        eta_late: Valor en épocas tardías

    Returns:
        η(a)
    """
    a_arr = np.asarray(a)
    x = (a_arr - params.a_trans) / max(params.delta_a, 1e-6)
    transition = 0.5 * (1.0 + np.tanh(x))
    return eta_early + (eta_late - eta_early) * transition


# =============================================================================
# Conexión con S
# =============================================================================

@dataclass
class MuEtaFromS:
    """Calcula μ y η desde el Campo de Adrián y el mapa entrópico.

    En el MCMC, las modificaciones de gravedad emergen de:
        μ(S) - 1 ∝ (dΦ_ten/dS)²  (contribución tensorial)
        η(S) - 1 ∝ V''(Φ_esc)    (contribución escalar)

    Attributes:
        s_map: Mapa entrópico
        alpha_mu: Acople para μ
        alpha_eta: Acople para η
    """
    s_map: EntropyMap | None = None
    alpha_mu: float = 0.01
    alpha_eta: float = 0.01

    def __post_init__(self):
        if self.s_map is None:
            self.s_map = EntropyMap()

    def mu_of_S(
        self,
        S: float | np.ndarray,
        dPhi_ten_dS: Callable[[float], float] | None = None
    ) -> float | np.ndarray:
        """μ(S) desde la derivada del campo tensorial.

        μ(S) = 1 + α_μ · (dΦ_ten/dS)²

        Args:
            S: Variable entrópica
            dPhi_ten_dS: Derivada dΦ_ten/dS (función)

        Returns:
            μ(S)
        """
        S_arr = np.asarray(S)

        if dPhi_ten_dS is None:
            # Usar aproximación del mapa
            phi_ten = self.s_map.phi_ten(S_arr)
            # Derivada numérica simple
            dS = 1e-5
            S_plus = S_arr + dS
            phi_ten_plus = self.s_map.phi_ten(S_plus)
            dPhi = (phi_ten_plus - phi_ten) / dS
        else:
            dPhi = np.vectorize(dPhi_ten_dS)(S_arr)

        return 1.0 + self.alpha_mu * dPhi**2

    def eta_of_S(
        self,
        S: float | np.ndarray,
        V_second: Callable[[float], float] | None = None
    ) -> float | np.ndarray:
        """η(S) desde la segunda derivada del potencial.

        CORRECCIÓN: S ∈ [0, 100], post-Big Bang S ∈ [1.001, 95.07]

        η(S) = 1 + α_η · V''(Φ_esc) / V_0

        Args:
            S: Variable entrópica
            V_second: V''(Φ_esc) normalizado

        Returns:
            η(S)
        """
        S_arr = np.asarray(S)

        if V_second is None:
            # Modelo simple: desviación pequeña cerca de transiciones
            # CORRECCIÓN: Transiciones en nuevo rango S ∈ [0, 100]
            S_BB = 1.001    # Big Bang
            S_peak = 47.5   # Pico formación estelar
            # Gaussianas con anchura proporcional al nuevo rango
            deviation = np.exp(-0.1 * (S_arr - S_BB)**2)
            deviation += np.exp(-0.01 * (S_arr - S_peak)**2)
            return 1.0 + self.alpha_eta * deviation

        V_pp = np.vectorize(V_second)(S_arr)
        return 1.0 + self.alpha_eta * V_pp

    def mu_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """μ(z) convertido desde S.

        Args:
            z: Redshift

        Returns:
            μ(z)
        """
        S = self.s_map.S_of_z(z)
        return self.mu_of_S(S)

    def eta_of_z(self, z: float | np.ndarray) -> float | np.ndarray:
        """η(z) convertido desde S.

        Args:
            z: Redshift

        Returns:
            η(z)
        """
        S = self.s_map.S_of_z(z)
        return self.eta_of_S(S)


# =============================================================================
# Observables combinados
# =============================================================================

def Sigma_lensing(
    mu: float | np.ndarray,
    eta: float | np.ndarray
) -> float | np.ndarray:
    """Parámetro de lensing Σ = μ·(1+η)/2.

    El lensing mide la combinación de potenciales Φ + Ψ.

    Args:
        mu: Parámetro μ
        eta: Parámetro η (slip)

    Returns:
        Σ para lensing
    """
    return mu * (1.0 + eta) / 2.0


def Upsilon_RSD(
    mu: float | np.ndarray,
    f: float | np.ndarray
) -> float | np.ndarray:
    """Parámetro RSD Υ = μ·f.

    Las distorsiones de espacio de redshift miden μ·f.

    Args:
        mu: Parámetro μ
        f: Tasa de crecimiento f = d ln D / d ln a

    Returns:
        Υ para RSD
    """
    return mu * f


def mu_eff_from_growth(
    f_obs: float,
    f_GR: float,
    Omega_m: float,
    gamma: float = 0.55
) -> float:
    """Estima μ_eff desde f observado vs GR.

    f_obs = f_GR · μ^(1/γ)  aproximadamente
    ⟹ μ ≈ (f_obs / f_GR)^γ

    Args:
        f_obs: Tasa de crecimiento observada
        f_GR: Tasa de crecimiento GR
        Omega_m: Fracción de materia
        gamma: Exponente de crecimiento

    Returns:
        μ_eff estimado
    """
    if f_GR < 1e-10:
        return 1.0
    ratio = f_obs / f_GR
    return ratio ** gamma


# =============================================================================
# Evolución perturbativa
# =============================================================================

@dataclass
class PerturbationParams:
    """Parámetros para evolución de perturbaciones.

    Attributes:
        k: Número de onda k [h/Mpc]
        mu_func: Función μ(a)
        eta_func: Función η(a)
        Omega_m0: Fracción de materia hoy
        H0: Constante de Hubble [km/s/Mpc]
    """
    k: float = 0.1  # h/Mpc
    mu_func: Callable[[float], float] | None = None
    eta_func: Callable[[float], float] | None = None
    Omega_m0: float = 0.315
    H0: float = 67.4

    def __post_init__(self):
        if self.mu_func is None:
            self.mu_func = lambda a: 1.0
        if self.eta_func is None:
            self.eta_func = lambda a: 1.0


def growth_ode_modified(
    a: float,
    y: np.ndarray,
    params: PerturbationParams
) -> np.ndarray:
    """ODE para crecimiento con gravedad modificada.

    δ'' + (2 + H'/H) δ' - (3/2) Ω_m(a) μ(a) δ = 0

    donde ' = d/d ln a.

    Variables: y = [δ, δ']

    Args:
        a: Factor de escala
        y: Vector [δ, δ']
        params: Parámetros

    Returns:
        dy/d(ln a)
    """
    delta, delta_prime = y

    # Omega_m(a)
    E2 = params.Omega_m0 * a**(-3) + (1 - params.Omega_m0)
    Omega_m = params.Omega_m0 * a**(-3) / E2

    # H'/H = d ln H / d ln a
    H_prime_over_H = -1.5 * Omega_m + (1 - Omega_m) * 0  # w_DE = -1

    # μ(a)
    mu = params.mu_func(a)

    # Ecuación de crecimiento modificada
    ddelta_prime = -(2.0 + H_prime_over_H) * delta_prime + 1.5 * Omega_m * mu * delta

    return np.array([delta_prime, ddelta_prime])


def solve_growth_modified(
    a_range: Tuple[float, float],
    params: PerturbationParams,
    n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resuelve crecimiento con gravedad modificada.

    Args:
        a_range: (a_min, a_max)
        params: Parámetros de perturbación
        n_points: Número de puntos

    Returns:
        (a_arr, D_arr, f_arr): Factor de escala, D(a), f(a)
    """
    from scipy.integrate import solve_ivp

    a_min, a_max = a_range

    # Condiciones iniciales en época de materia (D ∝ a)
    delta_init = a_min
    delta_prime_init = a_min  # d delta / d ln a = a en MD

    # Integrar en ln(a)
    ln_a = np.linspace(np.log(a_min), np.log(a_max), n_points)

    def ode_ln_a(ln_a_val, y):
        a_val = np.exp(ln_a_val)
        return growth_ode_modified(a_val, y, params)

    sol = solve_ivp(
        ode_ln_a,
        [ln_a[0], ln_a[-1]],
        [delta_init, delta_prime_init],
        t_eval=ln_a,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )

    a_arr = np.exp(sol.t)
    delta_arr = sol.y[0]
    delta_prime_arr = sol.y[1]

    # Normalizar D(a=1) = 1
    a1_idx = np.argmin(np.abs(a_arr - 1.0))
    D_arr = delta_arr / delta_arr[a1_idx]

    # f = d ln D / d ln a = delta' / delta
    f_arr = delta_prime_arr / np.maximum(delta_arr, 1e-30)

    return a_arr, D_arr, f_arr


# =============================================================================
# Comparación MCMC vs GR
# =============================================================================

@dataclass
class ModifiedGravityComparison:
    """Resultado de comparación gravedad modificada vs GR.

    Attributes:
        a: Array de factores de escala
        mu: μ(a)
        eta: η(a)
        Sigma: Σ para lensing
        D_mod: Factor de crecimiento modificado
        D_GR: Factor de crecimiento GR
        f_mod: Tasa de crecimiento modificada
        f_GR: Tasa de crecimiento GR
    """
    a: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    Sigma: np.ndarray
    D_mod: np.ndarray
    D_GR: np.ndarray
    f_mod: np.ndarray
    f_GR: np.ndarray


def compare_modified_gravity(
    mu_eta_params: MuEtaParams,
    Omega_m0: float = 0.315,
    a_range: Tuple[float, float] = (0.01, 1.0),
    n_points: int = 200
) -> ModifiedGravityComparison:
    """Compara evolución con gravedad modificada vs GR.

    Args:
        mu_eta_params: Parámetros de μ y η
        Omega_m0: Fracción de materia
        a_range: Rango de factor de escala
        n_points: Número de puntos

    Returns:
        ModifiedGravityComparison
    """
    a = np.linspace(a_range[0], a_range[1], n_points)

    # μ y η
    mu = mu_CPL(a, mu_eta_params)
    eta = eta_CPL(a, mu_eta_params)
    Sigma = Sigma_lensing(mu, eta)

    # Crecimiento GR
    params_GR = PerturbationParams(
        mu_func=lambda x: 1.0,
        eta_func=lambda x: 1.0,
        Omega_m0=Omega_m0,
    )
    _, D_GR, f_GR = solve_growth_modified(a_range, params_GR, n_points)

    # Crecimiento modificado
    params_mod = PerturbationParams(
        mu_func=lambda x: mu_CPL(x, mu_eta_params),
        eta_func=lambda x: eta_CPL(x, mu_eta_params),
        Omega_m0=Omega_m0,
    )
    _, D_mod, f_mod = solve_growth_modified(a_range, params_mod, n_points)

    return ModifiedGravityComparison(
        a=a,
        mu=mu,
        eta=eta,
        Sigma=Sigma,
        D_mod=D_mod,
        D_GR=D_GR,
        f_mod=f_mod,
        f_GR=f_GR,
    )
