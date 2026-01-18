"""
Canal Indeterminado ρ_id - Modelo Ontológico (Nivel B)
======================================================

Este módulo implementa el canal de energía indeterminada ρ_id usando
el formalismo ontológico completo del MCMC, con balances en la
variable entrópica S.

Las ecuaciones de balance en S son:
    dρ_id/dS = -γ * ρ_id + δ(S)

donde:
- γ: tasa de "conversión" o dilución
- δ(S): función fuente que depende de la conversión masa→espacio

Este nivel es más completo que el paramétrico y permite:
- Reconstrucción de w_DE(z)
- Conexión con el Campo de Adrián
- Restricciones físicas desde primeros principios

Referencias:
    - Tratado MCMC Maestro: ecuaciones de balance
    - Bloque 1: canal indeterminado/vacío cuántico emergente
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.interpolate import CubicSpline


@dataclass
class RhoIdOntologicalParams:
    """
    Parámetros del modelo ontológico de ρ_id.

    Estos parámetros controlan las ecuaciones de balance dρ_id/dS = -γ*ρ_id + δ(S).

    Attributes:
        rho_id_init: Valor inicial de ρ_id en S_min (condición de frontera)
        gamma: Tasa de dilución/conversión
        delta_amplitude: Amplitud de la función fuente δ(S)
        delta_S_peak: Posición del pico de la fuente (en S)
        delta_width: Ancho del pico de la fuente
        coupling_phi: Acoplamiento con el campo tensional Φ_ten
    """
    rho_id_init: float = 0.01       # Condición inicial (pequeña en S temprano)
    gamma: float = 1.0              # Tasa de dilución
    delta_amplitude: float = 1.0    # Amplitud de la fuente
    delta_S_peak: float = 0.5       # Centro del pico de producción
    delta_width: float = 0.2        # Ancho del pico
    coupling_phi: float = 0.1       # Acoplamiento con Φ_ten

    def to_dict(self) -> Dict[str, float]:
        """Convierte a diccionario."""
        return vars(self).copy()

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'RhoIdOntologicalParams':
        """Construye desde diccionario."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def delta_source_gaussian(
    S: np.ndarray,
    params: RhoIdOntologicalParams
) -> np.ndarray:
    """
    Función fuente δ(S) con forma gaussiana.

    Representa la "producción" de ρ_id desde la conversión masa→espacio,
    concentrada en una región de S.

    Args:
        S: Array de variable entrópica
        params: Parámetros del modelo

    Returns:
        Array con δ(S)
    """
    return params.delta_amplitude * np.exp(
        -0.5 * ((S - params.delta_S_peak) / params.delta_width)**2
    )


def delta_source_with_phi(
    S: np.ndarray,
    params: RhoIdOntologicalParams,
    Phi_ten: np.ndarray
) -> np.ndarray:
    """
    Función fuente δ(S) acoplada al campo tensional.

    δ(S) = A * exp[...] * (1 + κ * Φ_ten(S))

    El acoplamiento con Φ_ten representa la conexión física entre
    el campo de Adrián y la producción de vacío cuántico emergente.

    Args:
        S: Array de variable entrópica
        params: Parámetros del modelo
        Phi_ten: Campo tensional

    Returns:
        Array con δ(S)
    """
    delta_base = delta_source_gaussian(S, params)
    coupling_factor = 1.0 + params.coupling_phi * Phi_ten
    return delta_base * coupling_factor


def solve_rho_id_balance(
    S: np.ndarray,
    params: RhoIdOntologicalParams,
    Phi_ten: Optional[np.ndarray] = None,
    method: str = 'cumulative'
) -> np.ndarray:
    """
    Resuelve la ecuación de balance para ρ_id en S.

    dρ_id/dS = -γ * ρ_id + δ(S)

    Esta es una ODE lineal de primer orden que se resuelve
    hacia adelante desde S_min.

    Args:
        S: Array de variable entrópica (ordenado creciente)
        params: Parámetros del modelo
        Phi_ten: Campo tensional (opcional)
        method: Método de integración ('cumulative' o 'ivp')

    Returns:
        Array con ρ_id(S)
    """
    # Calcular la fuente
    if Phi_ten is not None:
        delta = delta_source_with_phi(S, params, Phi_ten)
    else:
        delta = delta_source_gaussian(S, params)

    if method == 'ivp':
        return _solve_rho_id_ivp(S, params, delta)
    else:
        return _solve_rho_id_cumulative(S, params, delta)


def _solve_rho_id_cumulative(
    S: np.ndarray,
    params: RhoIdOntologicalParams,
    delta: np.ndarray
) -> np.ndarray:
    """
    Resuelve la ecuación de balance usando integración cumulativa.

    La solución analítica de dρ/dS = -γρ + δ es:
    ρ(S) = ρ_0 * exp(-γS) + exp(-γS) * ∫₀^S exp(γS') δ(S') dS'

    Args:
        S: Rejilla entrópica
        params: Parámetros
        delta: Función fuente

    Returns:
        Array con ρ_id(S)
    """
    gamma = params.gamma

    # Factor exponencial
    exp_minus_gamma_S = np.exp(-gamma * (S - S[0]))

    # Término homogéneo
    rho_homogeneous = params.rho_id_init * exp_minus_gamma_S

    # Integral del término particular
    # ∫₀^S exp(γS') δ(S') dS'
    integrand = np.exp(gamma * (S - S[0])) * delta
    integral = np.zeros_like(S)
    integral[1:] = cumulative_trapezoid(integrand, S)

    # Solución completa
    rho_id = rho_homogeneous + exp_minus_gamma_S * integral

    return rho_id


def _solve_rho_id_ivp(
    S: np.ndarray,
    params: RhoIdOntologicalParams,
    delta: np.ndarray
) -> np.ndarray:
    """
    Resuelve la ecuación de balance usando solve_ivp.

    Args:
        S: Rejilla entrópica
        params: Parámetros
        delta: Función fuente

    Returns:
        Array con ρ_id(S)
    """
    # Interpolador para δ(S)
    delta_interp = CubicSpline(S, delta)

    def rhs(s, rho):
        return -params.gamma * rho + delta_interp(s)

    # Resolver ODE
    sol = solve_ivp(
        rhs,
        (S[0], S[-1]),
        [params.rho_id_init],
        t_eval=S,
        method='RK45'
    )

    return sol.y[0]


def pressure_id_from_rho(
    rho_id: np.ndarray,
    w_id: np.ndarray
) -> np.ndarray:
    """
    Calcula la presión p_id = w_id * ρ_id.

    Args:
        rho_id: Densidad de energía
        w_id: Ecuación de estado

    Returns:
        Array con p_id(S)
    """
    return w_id * rho_id


def reconstruct_w_id_from_evolution(
    S: np.ndarray,
    rho_id: np.ndarray,
    a: np.ndarray
) -> np.ndarray:
    """
    Reconstruye w_id(S) desde la evolución de ρ_id.

    Usando la ecuación de conservación:
    dρ/dt + 3H(ρ + p) = 0  =>  w = -1 - (1/3) * d ln ρ / d ln a

    Args:
        S: Rejilla entrópica
        rho_id: Densidad de ρ_id
        a: Factor de escala

    Returns:
        Array con w_id(S)
    """
    # d ln ρ / d ln a usando derivación numérica
    ln_rho = np.log(np.maximum(rho_id, 1e-30))  # Evitar log(0)
    ln_a = np.log(a)

    # Derivada numérica
    d_ln_rho = np.gradient(ln_rho, S)
    d_ln_a = np.gradient(ln_a, S)

    # Evitar división por cero
    mask = np.abs(d_ln_a) > 1e-10
    d_ln_rho_d_ln_a = np.zeros_like(S)
    d_ln_rho_d_ln_a[mask] = d_ln_rho[mask] / d_ln_a[mask]

    # w = -1 - (1/3) * d ln ρ / d ln a
    w_id = -1.0 - d_ln_rho_d_ln_a / 3.0

    return w_id


class RhoIdOntologicalModel:
    """
    Clase que encapsula el modelo ontológico de ρ_id.

    Este modelo resuelve las ecuaciones de balance en S y
    reconstruye w_DE(z) desde primeros principios.
    """

    def __init__(
        self,
        S: np.ndarray,
        params: RhoIdOntologicalParams,
        Phi_ten: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None
    ):
        """
        Inicializa y resuelve el modelo.

        Args:
            S: Rejilla entrópica
            params: Parámetros del modelo
            Phi_ten: Campo tensional (opcional)
            a: Factor de escala (opcional, para reconstrucción de w)
            z: Redshift (opcional)
        """
        self.S = S
        self.params = params
        self.Phi_ten = Phi_ten
        self.a = a
        self.z = z

        # Resolver la ecuación de balance
        self.rho_id = solve_rho_id_balance(S, params, Phi_ten)

        # Calcular la fuente
        if Phi_ten is not None:
            self.delta = delta_source_with_phi(S, params, Phi_ten)
        else:
            self.delta = delta_source_gaussian(S, params)

        # Reconstruir w si tenemos a(S)
        if a is not None:
            self.w_id = reconstruct_w_id_from_evolution(S, self.rho_id, a)
        else:
            self.w_id = None

        # Construir interpoladores
        self._build_interpolators()

    def _build_interpolators(self):
        """Construye interpoladores para evaluación eficiente."""
        self._rho_id_of_S = CubicSpline(self.S, self.rho_id)

        if self.z is not None:
            # Ordenar por z creciente
            idx = np.argsort(self.z)
            self._rho_id_of_z = CubicSpline(self.z[idx], self.rho_id[idx])

            if self.w_id is not None:
                self._w_id_of_z = CubicSpline(self.z[idx], self.w_id[idx])

    def rho_id_at_S(self, S: float) -> float:
        """Evalúa ρ_id en S dado."""
        return float(self._rho_id_of_S(S))

    def rho_id_at_z(self, z: float) -> float:
        """Evalúa ρ_id en z dado."""
        if self.z is None:
            raise ValueError("z no fue proporcionado en la inicialización")
        return float(self._rho_id_of_z(z))

    def w_id_at_z(self, z: float) -> float:
        """Evalúa w_id en z dado."""
        if self.w_id is None:
            raise ValueError("a no fue proporcionado, no se puede calcular w_id")
        return float(self._w_id_of_z(z))

    @property
    def Omega_id_today(self) -> float:
        """Fracción de ρ_id hoy (en S_max)."""
        return self.rho_id[-1]


# Exportaciones
__all__ = [
    'RhoIdOntologicalParams',
    'delta_source_gaussian',
    'delta_source_with_phi',
    'solve_rho_id_balance',
    'pressure_id_from_rho',
    'reconstruct_w_id_from_evolution',
    'RhoIdOntologicalModel',
]
