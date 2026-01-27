"""Integrador N-body en variable entrópica S.

El integrador N-body Cronos utiliza S como variable de integración
fundamental, siguiendo la Ley de Cronos. El algoritmo Leapfrog KDK
se modifica para incluir:

    1. Dilatación temporal por densidad local
    2. Fuerza adicional de ρ_id
    3. Paso en S en lugar de t

Relación fundamental:
    dt_rel/dS = (λ_C/k_α) · tanh(S/λ_C)  [Ley de Cronos]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .kdk import State
from .poisson import estimate_density


@dataclass
class CronosIntegratorParams:
    """Parámetros del integrador Cronos en variable S.

    Attributes:
        S_ini: S inicial (típico 1.0005 para z_ini ~ 100)
        S_fin: S final (típico 1.001 para z ~ 0)
        dS: Paso en S
        lambda_C: Parámetro de la Ley de Cronos
        k_alpha: Factor de escala temporal
        eta: Factor de precisión (Courant)
        zeta_0: Fuerza de dilatación temporal
        rho_star: Densidad de saturación
        alpha_tens: Coeficiente de Φ_ten
    """
    S_ini: float = 1.0005
    S_fin: float = 1.001
    dS: float = 1e-5
    lambda_C: float = 0.001
    k_alpha: float = 1.0
    eta: float = 0.02
    zeta_0: float = 0.02
    rho_star: float = 1.0
    alpha_tens: float = 0.1


@dataclass
class CronosState:
    """Estado del sistema en el integrador Cronos.

    Attributes:
        S: Variable entrópica actual
        x: Posiciones (N, 3)
        v: Velocidades (N, 3)
        m: Masas (N,)
        rho: Densidad local (N,)
        a_scale: Factor de escala cosmológico
    """
    S: float
    x: np.ndarray
    v: np.ndarray
    m: np.ndarray | None = None
    rho: np.ndarray = field(default_factory=lambda: np.array([]))
    a_scale: float = 1.0

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        if self.m is not None:
            self.m = np.asarray(self.m, dtype=float)

    @property
    def N(self) -> int:
        return self.x.shape[0]

    def to_standard_state(self) -> State:
        """Convierte a State estándar para compatibilidad."""
        return State(x=self.x, v=self.v, m=self.m)


def dt_of_dS(S: float, params: CronosIntegratorParams) -> float:
    """Calcula dt/dS usando la Ley de Cronos.

    dt_rel/dS = (λ_C/k_α) · tanh(S/λ_C)

    Args:
        S: Variable entrópica
        params: Parámetros del integrador

    Returns:
        dt/dS en unidades de código
    """
    arg = S / max(params.lambda_C, 1e-12)
    return (params.lambda_C / params.k_alpha) * np.tanh(arg)


def a_scale_of_S(
    S: float,
    S_BB: float = 1.001,
    a_BB: float = 1.0
) -> float:
    """Factor de escala cosmológico en función de S.

    Aproximación simple: a(S) ∝ S para S cercano a S_BB.
    En una implementación completa, esto vendría de resolver
    las ecuaciones de Friedmann modificadas.

    Args:
        S: Variable entrópica
        S_BB: S del Big Bang
        a_BB: Factor de escala en S_BB (normalización)

    Returns:
        Factor de escala a(S)
    """
    # Modelo simple: expansión lineal cerca de S_BB
    return a_BB * (S / S_BB)


def cronos_kick_step(
    state: CronosState,
    acc_fn: Callable[[np.ndarray], np.ndarray],
    params: CronosIntegratorParams,
    rho_id_fn: Callable[[np.ndarray, float], np.ndarray] | None = None
) -> CronosState:
    """Paso Kick modificado con dilatación Cronos.

    El kick incluye:
    1. Aceleración gravitatoria estándar
    2. Contribución de ρ_id (si se proporciona)
    3. Dilatación temporal local por densidad

    Args:
        state: Estado actual
        acc_fn: Función de aceleración gravitatoria
        params: Parámetros del integrador
        rho_id_fn: Función que da aceleración extra de ρ_id (opcional)

    Returns:
        Estado con velocidades actualizadas (medio paso)
    """
    a = state.a_scale
    rho_c = params.rho_star

    # 1. Aceleración Newtoniana
    acc_newton = acc_fn(state.x)

    # 2. Contribución de ρ_id (potencial extra)
    if rho_id_fn is not None:
        acc_id = rho_id_fn(state.x, state.S)
    else:
        acc_id = np.zeros_like(acc_newton)

    # 3. Aceleración total
    acc_total = acc_newton + acc_id

    # 4. Calcular dt local con dilatación Cronos
    acc_norm = np.linalg.norm(acc_total, axis=1)
    acc_norm_safe = np.maximum(acc_norm, 1e-18)

    # Paso base por criterio de aceleración
    dt_base = params.eta / np.sqrt(acc_norm_safe * max(a, 1e-12))

    # Factor de dilatación por densidad local
    # [1 + (ρ_i/ρ_c)^(3/2) / α]^{-1}
    rho_ratio = state.rho / max(rho_c, 1e-12)
    dilation_factor = 1.0 / (1.0 + rho_ratio**1.5 / max(params.alpha_tens, 1e-12))

    dt_local = dt_base * dilation_factor

    # 5. Paso global (conservador)
    dt = params.zeta_0 * np.min(dt_local)

    # 6. Actualizar velocidades (medio paso)
    v_new = state.v + 0.5 * dt * acc_total

    return CronosState(
        S=state.S,
        x=state.x,
        v=v_new,
        m=state.m,
        rho=state.rho,
        a_scale=state.a_scale,
    )


def cronos_drift_step(
    state: CronosState,
    dS: float,
    params: CronosIntegratorParams
) -> CronosState:
    """Paso Drift en variable S.

    Avanza posiciones usando dt = (dt/dS) · dS

    Args:
        state: Estado actual
        dS: Paso en S
        params: Parámetros del integrador

    Returns:
        Estado con posiciones y S actualizados
    """
    # dt correspondiente a dS
    dt = dt_of_dS(state.S, params) * dS

    # Actualizar posiciones
    x_new = state.x + dt * state.v

    # Nuevo S y factor de escala
    S_new = state.S + dS
    a_new = a_scale_of_S(S_new)

    return CronosState(
        S=S_new,
        x=x_new,
        v=state.v,
        m=state.m,
        rho=state.rho,
        a_scale=a_new,
    )


def update_density(
    state: CronosState,
    h: float
) -> CronosState:
    """Actualiza estimación de densidad local.

    Args:
        state: Estado actual
        h: Radio del kernel SPH

    Returns:
        Estado con densidades actualizadas
    """
    rho_new = estimate_density(state.x, state.m, h)

    return CronosState(
        S=state.S,
        x=state.x,
        v=state.v,
        m=state.m,
        rho=rho_new,
        a_scale=state.a_scale,
    )


def cronos_kdk_step(
    state: CronosState,
    acc_fn: Callable[[np.ndarray], np.ndarray],
    params: CronosIntegratorParams,
    dS: float,
    rho_id_fn: Callable[[np.ndarray, float], np.ndarray] | None = None
) -> CronosState:
    """Paso KDK completo en variable S.

    1. Kick (medio paso)
    2. Drift (paso completo en S)
    3. Kick (medio paso)

    Args:
        state: Estado inicial
        acc_fn: Función de aceleración
        params: Parámetros del integrador
        dS: Paso en S
        rho_id_fn: Función de aceleración ρ_id (opcional)

    Returns:
        Estado después de un paso dS
    """
    # Kick 1
    state = cronos_kick_step(state, acc_fn, params, rho_id_fn)

    # Drift
    state = cronos_drift_step(state, dS, params)

    # Kick 2
    state = cronos_kick_step(state, acc_fn, params, rho_id_fn)

    return state


def integrate_cronos(
    initial_state: CronosState,
    acc_fn: Callable[[np.ndarray], np.ndarray],
    params: CronosIntegratorParams,
    save_every: int = 100,
    update_density_every: int = 10,
    density_kernel_h: float = 1.0,
    rho_id_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    callback: Callable[[CronosState, int], None] | None = None
) -> list[CronosState]:
    """Integra el sistema N-body en variable S.

    Args:
        initial_state: Estado inicial
        acc_fn: Función de aceleración gravitatoria
        params: Parámetros del integrador
        save_every: Guardar estado cada N pasos
        update_density_every: Actualizar densidad cada N pasos
        density_kernel_h: Radio del kernel para densidad
        rho_id_fn: Función de aceleración ρ_id
        callback: Función llamada en cada paso (para diagnósticos)

    Returns:
        Lista de estados guardados
    """
    # Calcular número de pasos
    n_steps = int((params.S_fin - params.S_ini) / params.dS)
    dS = params.dS

    states = [initial_state]
    state = initial_state

    # Inicializar densidad
    state = update_density(state, density_kernel_h)

    for i in range(n_steps):
        # Actualizar densidad periódicamente
        if i % update_density_every == 0:
            state = update_density(state, density_kernel_h)

        # Paso KDK
        state = cronos_kdk_step(state, acc_fn, params, dS, rho_id_fn)

        # Callback opcional
        if callback is not None:
            callback(state, i)

        # Guardar estado
        if (i + 1) % save_every == 0:
            states.append(state)

    # Guardar estado final
    if n_steps % save_every != 0:
        states.append(state)

    return states


# ---------------------------------------------------------------------
# Generador de condiciones iniciales
# ---------------------------------------------------------------------

def generate_uniform_ic(
    N: int,
    L_box: float,
    seed: int = 42
) -> CronosState:
    """Genera condiciones iniciales uniformes (para testing).

    Args:
        N: Número de partículas
        L_box: Tamaño de la caja
        seed: Semilla aleatoria

    Returns:
        Estado inicial uniforme
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, L_box, size=(N, 3))
    v = np.zeros((N, 3))  # Partículas en reposo
    m = np.ones(N) / N    # Masa total normalizada

    return CronosState(
        S=1.0005,  # z ~ 100
        x=x,
        v=v,
        m=m,
        rho=np.ones(N),
        a_scale=0.01,  # a ~ 0.01 para z ~ 100
    )


def generate_perturbation_ic(
    N: int,
    L_box: float,
    P_k: Callable[[np.ndarray], np.ndarray],
    seed: int = 42
) -> CronosState:
    """Genera condiciones iniciales con perturbaciones P(k).

    Implementación simplificada usando el método de Zel'dovich.

    Args:
        N: Número de partículas (debe ser cubo perfecto)
        L_box: Tamaño de la caja
        P_k: Espectro de potencia P(k)
        seed: Semilla aleatoria

    Returns:
        Estado inicial con perturbaciones
    """
    rng = np.random.default_rng(seed)

    # Grid regular
    N_side = int(round(N ** (1/3)))
    if N_side ** 3 != N:
        raise ValueError(f"N={N} debe ser un cubo perfecto")

    # Posiciones en grid
    idx = np.arange(N_side)
    xx, yy, zz = np.meshgrid(idx, idx, idx, indexing='ij')
    x_grid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    x_grid = x_grid.astype(float) * (L_box / N_side)

    # Generar campo de desplazamiento Gaussiano (simplificado)
    # En una implementación completa, esto usaría FFT y P(k)
    displacement = rng.normal(0, 0.01 * L_box, size=(N, 3))

    # Aplicar desplazamiento
    x = x_grid + displacement
    x = x % L_box  # Condiciones periódicas

    # Velocidades iniciales (aproximación de Zel'dovich)
    H_ini = 100.0  # H(z=100) aproximado
    v = H_ini * displacement * 0.01  # Escalar apropiadamente

    m = np.ones(N) / N

    return CronosState(
        S=1.0005,
        x=x,
        v=v,
        m=m,
        rho=np.ones(N),
        a_scale=0.01,
    )
