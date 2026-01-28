"""Algoritmos Monte Carlo para simulaciones lattice-gauge.

Implementa:
    - Metropolis: Actualización local con aceptación/rechazo
    - Heatbath: Actualización local exacta (para U(1))

Ambos algoritmos generan una cadena de configuraciones gauge
muestreando la distribución de Boltzmann exp(-S[U]).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .wilson import WilsonAction


@dataclass
class MCParams:
    """Parámetros para Monte Carlo.

    Attributes:
        n_therm: Pasos de termalización
        n_conf: Número de configuraciones a generar
        n_skip: Pasos entre configuraciones (decorrelación)
        delta: Tamaño del paso Metropolis
        seed: Semilla aleatoria
    """
    n_therm: int = 1000
    n_conf: int = 100
    n_skip: int = 10
    delta: float = 0.5
    seed: int = 42


class MetropolisSampler:
    """Muestreador Metropolis para teoría gauge.

    El algoritmo Metropolis actualiza enlaces individuales
    con probabilidad de aceptación:

        P_acc = min(1, exp(-ΔS))

    donde ΔS es el cambio en la acción.

    Attributes:
        action: Acción de Wilson
        params: Parámetros MC
        rng: Generador de números aleatorios
        acceptance_rate: Tasa de aceptación acumulada
    """

    def __init__(self, action: WilsonAction, params: MCParams):
        """Inicializa el muestreador.

        Args:
            action: Acción de Wilson configurada
            params: Parámetros MC
        """
        self.action = action
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self._n_accepted = 0
        self._n_proposed = 0

    @property
    def acceptance_rate(self) -> float:
        """Tasa de aceptación actual."""
        if self._n_proposed == 0:
            return 0.0
        return self._n_accepted / self._n_proposed

    def _metropolis_update(self, site: tuple, mu: int) -> bool:
        """Propone y acepta/rechaza actualización de un enlace.

        Args:
            site: Sitio del enlace
            mu: Dirección

        Returns:
            True si se aceptó la propuesta
        """
        # Acción local antes
        S_old = self.action.local_action(site, mu)

        # Proponer nuevo valor
        old_link = self.action._get_link(site, mu)
        new_link = old_link + self.rng.uniform(-self.params.delta, self.params.delta)

        # Aplicar propuesta
        self.action._set_link(site, mu, new_link)

        # Acción local después
        S_new = self.action.local_action(site, mu)

        # Criterio de Metropolis
        delta_S = S_new - S_old
        self._n_proposed += 1

        if delta_S <= 0 or self.rng.random() < np.exp(-delta_S):
            # Aceptar
            self._n_accepted += 1
            return True
        else:
            # Rechazar: restaurar valor anterior
            self.action._set_link(site, mu, old_link)
            return False

    def sweep(self):
        """Realiza un barrido completo de la retícula.

        Actualiza todos los enlaces una vez en orden lexicográfico.
        """
        p = self.action.params

        for idx in np.ndindex(*([p.L] * p.n_dim)):
            for mu in range(p.n_dim):
                self._metropolis_update(idx, mu)

    def thermalize(self, n_sweeps: int | None = None):
        """Termaliza la configuración.

        Args:
            n_sweeps: Número de barridos (default: params.n_therm)
        """
        if n_sweeps is None:
            n_sweeps = self.params.n_therm

        for _ in range(n_sweeps):
            self.sweep()

    def generate_configurations(
        self,
        n_conf: int | None = None,
        n_skip: int | None = None,
        thermalize: bool = True,
        callback: Callable[[int, WilsonAction], None] | None = None
    ) -> list[np.ndarray]:
        """Genera configuraciones gauge.

        Args:
            n_conf: Número de configuraciones
            n_skip: Barridos entre configuraciones
            thermalize: Si True, termaliza primero
            callback: Función llamada en cada configuración

        Returns:
            Lista de configuraciones (copias de los enlaces)
        """
        if n_conf is None:
            n_conf = self.params.n_conf
        if n_skip is None:
            n_skip = self.params.n_skip

        # Termalizar
        if thermalize:
            self.thermalize()

        # Generar configuraciones
        configurations = []

        for i in range(n_conf):
            # Decorrelación
            for _ in range(n_skip):
                self.sweep()

            # Guardar configuración
            configurations.append(self.action.links.copy())

            # Callback opcional
            if callback is not None:
                callback(i, self.action)

        return configurations

    def reset_statistics(self):
        """Reinicia contadores de aceptación."""
        self._n_accepted = 0
        self._n_proposed = 0


class HeatbathSampler:
    """Muestreador Heatbath para U(1).

    Para teoría U(1), el enlace puede muestrearse exactamente
    de la distribución condicional usando el método de rechazo.

    La distribución es: P(θ) ∝ exp(β·|S|·cos(θ - arg(S)))
    donde S es la suma de staples.

    Attributes:
        action: Acción de Wilson
        params: Parámetros MC
        rng: Generador aleatorio
    """

    def __init__(self, action: WilsonAction, params: MCParams):
        """Inicializa el muestreador heatbath."""
        self.action = action
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def _heatbath_update(self, site: tuple, mu: int):
        """Actualiza un enlace usando heatbath.

        Para U(1), muestreamos de la distribución de von Mises:
        P(θ) ∝ exp(κ·cos(θ - μ))

        Args:
            site: Sitio del enlace
            mu: Dirección
        """
        beta = self.action.params.beta

        # Suma de staples (número complejo)
        staple = self.action.staple_sum(site, mu)
        staple_abs = np.abs(staple)
        staple_arg = np.angle(staple) if staple_abs > 1e-10 else 0.0

        # Parámetro de concentración
        kappa = beta * staple_abs

        if kappa < 0.01:
            # Distribución casi uniforme
            new_angle = self.rng.uniform(-np.pi, np.pi)
        else:
            # Muestreo de von Mises usando algoritmo de Best-Fisher
            new_angle = self._sample_von_mises(kappa) + staple_arg

        self.action._set_link(site, mu, new_angle)

    def _sample_von_mises(self, kappa: float) -> float:
        """Muestrea de distribución von Mises centrada en 0.

        Usa el algoritmo de Best-Fisher para κ > 0.

        Args:
            kappa: Parámetro de concentración

        Returns:
            Ángulo muestreado
        """
        # Algoritmo simplificado para κ moderado
        tau = 1.0 + np.sqrt(1.0 + 4.0 * kappa**2)
        rho = (tau - np.sqrt(2.0 * tau)) / (2.0 * kappa)
        r = (1.0 + rho**2) / (2.0 * rho)

        while True:
            u1 = self.rng.random()
            z = np.cos(np.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)

            u2 = self.rng.random()
            if c * (2.0 - c) > u2 or np.log(c / u2) + 1.0 >= c:
                u3 = self.rng.random()
                return np.arccos(f) * (1 if u3 < 0.5 else -1)

    def sweep(self):
        """Realiza un barrido completo con heatbath."""
        p = self.action.params

        for idx in np.ndindex(*([p.L] * p.n_dim)):
            for mu in range(p.n_dim):
                self._heatbath_update(idx, mu)

    def thermalize(self, n_sweeps: int | None = None):
        """Termaliza la configuración."""
        if n_sweeps is None:
            n_sweeps = self.params.n_therm

        for _ in range(n_sweeps):
            self.sweep()

    def generate_configurations(
        self,
        n_conf: int | None = None,
        n_skip: int | None = None,
        thermalize: bool = True
    ) -> list[np.ndarray]:
        """Genera configuraciones gauge usando heatbath."""
        if n_conf is None:
            n_conf = self.params.n_conf
        if n_skip is None:
            n_skip = self.params.n_skip

        if thermalize:
            self.thermalize()

        configurations = []

        for _ in range(n_conf):
            for _ in range(n_skip):
                self.sweep()
            configurations.append(self.action.links.copy())

        return configurations


def autocorrelation_time(observable: np.ndarray, max_lag: int = 100) -> float:
    """Estima el tiempo de autocorrelación integrado.

    τ_int = 1/2 + Σ_{t=1}^{∞} C(t)/C(0)

    Args:
        observable: Serie temporal del observable
        max_lag: Máximo lag a considerar

    Returns:
        Tiempo de autocorrelación integrado
    """
    n = len(observable)
    if n < 2:
        return 1.0

    mean = np.mean(observable)
    var = np.var(observable)

    if var < 1e-15:
        return 1.0

    # Función de autocorrelación
    tau_int = 0.5

    for t in range(1, min(max_lag, n // 2)):
        # C(t) = ⟨(x_i - μ)(x_{i+t} - μ)⟩
        c_t = np.mean((observable[:-t] - mean) * (observable[t:] - mean))
        rho_t = c_t / var

        if rho_t < 0.05:  # Cortar cuando es despreciable
            break

        tau_int += rho_t

    return max(tau_int, 1.0)
