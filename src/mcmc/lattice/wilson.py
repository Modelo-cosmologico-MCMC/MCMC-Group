"""Acción de Wilson para teoría gauge en retícula.

La acción de Wilson es la discretización estándar de Yang-Mills:

    S_YM[U] = β Σ_□ [1 - (1/N) Re Tr U_□]

donde U_□ son los plaquettes, β = 2N/g² es el acoplo, y la suma
recorre todos los plaquettes de la retícula L⁴.

En el MCMC se promueve el acoplo a función de S:
    β → β(S) = β₀ + β₁ · exp[-b_S · (S - S₃)]
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class WilsonActionParams:
    """Parámetros de la acción de Wilson.

    Attributes:
        beta: Constante de acoplo β = 2N/g²
        N_color: Número de colores (3 para QCD)
        L: Tamaño de retícula en cada dimensión
        n_dim: Número de dimensiones (4 para 4D)
    """
    beta: float = 6.0
    N_color: int = 3
    L: int = 8
    n_dim: int = 4


class WilsonAction:
    """Acción de Wilson para teoría gauge SU(N).

    Implementación simplificada usando U(1) como aproximación
    para demostraciones. Una implementación completa requeriría
    matrices SU(N) explícitas.

    Attributes:
        params: Parámetros de la acción
        links: Campo gauge (enlaces)
    """

    def __init__(self, params: WilsonActionParams):
        """Inicializa la acción de Wilson.

        Args:
            params: Parámetros de configuración
        """
        self.params = params
        self._init_links()

    def _init_links(self):
        """Inicializa los enlaces gauge a la identidad (cold start)."""
        p = self.params
        # Shape: (L, L, L, L, n_dim) para enlaces
        # Cada enlace es un ángulo (aproximación U(1))
        shape = tuple([p.L] * p.n_dim) + (p.n_dim,)
        self.links = np.zeros(shape, dtype=float)

    def hot_start(self, seed: int = 42):
        """Inicializa con configuración aleatoria (hot start)."""
        rng = np.random.default_rng(seed)
        self.links = rng.uniform(-np.pi, np.pi, size=self.links.shape)

    def cold_start(self):
        """Inicializa con todos los enlaces = 0 (identidad)."""
        self.links[:] = 0.0

    def _get_link(self, site: tuple, mu: int) -> float:
        """Obtiene el enlace U_μ(x) en el sitio dado.

        Args:
            site: Coordenadas del sitio (x₀, x₁, x₂, x₃)
            mu: Dirección del enlace (0, 1, 2, 3)

        Returns:
            Ángulo del enlace (aproximación U(1))
        """
        return self.links[site][mu]

    def _set_link(self, site: tuple, mu: int, value: float):
        """Establece el enlace U_μ(x)."""
        self.links[site][mu] = value

    def _plaquette(self, site: tuple, mu: int, nu: int) -> float:
        """Calcula el plaquette U_□(x; μ,ν).

        U_□ = U_μ(x) U_ν(x+μ̂) U†_μ(x+ν̂) U†_ν(x)

        En aproximación U(1): U_□ = exp(i θ_□)

        Args:
            site: Sitio base
            mu, nu: Direcciones del plaquette

        Returns:
            Re Tr U_□ (normalizado por N para U(1))
        """
        L = self.params.L

        # Sitios vecinos con condiciones periódicas
        site_mu = list(site)
        site_mu[mu] = (site_mu[mu] + 1) % L
        site_mu = tuple(site_mu)

        site_nu = list(site)
        site_nu[nu] = (site_nu[nu] + 1) % L
        site_nu = tuple(site_nu)

        # Enlaces del plaquette
        U1 = self._get_link(site, mu)
        U2 = self._get_link(site_mu, nu)
        U3 = self._get_link(site_nu, mu)  # Dagger = negativo para U(1)
        U4 = self._get_link(site, nu)     # Dagger

        # Suma de ángulos (aproximación U(1))
        theta_plaq = U1 + U2 - U3 - U4

        # Re Tr U_□ / N ≈ cos(θ_□) para U(1)
        return np.cos(theta_plaq)

    def action(self) -> float:
        """Calcula la acción total.

        S = β Σ_□ [1 - (1/N) Re Tr U_□]

        Returns:
            Valor de la acción
        """
        p = self.params
        total = 0.0

        # Iterar sobre todos los sitios
        for idx in np.ndindex(*([p.L] * p.n_dim)):
            # Iterar sobre todos los pares de direcciones
            for mu in range(p.n_dim):
                for nu in range(mu + 1, p.n_dim):
                    plaq = self._plaquette(idx, mu, nu)
                    total += 1.0 - plaq

        return p.beta * total

    def local_action(self, site: tuple, mu: int) -> float:
        """Calcula la acción local que involucra un enlace específico.

        Útil para actualizaciones de Metropolis.

        Args:
            site: Sitio del enlace
            mu: Dirección del enlace

        Returns:
            Contribución a la acción del enlace dado
        """
        p = self.params
        L = p.L
        total = 0.0

        # Plaquettes que contienen este enlace
        for nu in range(p.n_dim):
            if nu == mu:
                continue

            # Plaquette en el plano (μ, ν) con base en site
            total += 1.0 - self._plaquette(site, mu, nu)

            # Plaquette con base en site - ν̂
            site_minus_nu = list(site)
            site_minus_nu[nu] = (site_minus_nu[nu] - 1) % L
            site_minus_nu = tuple(site_minus_nu)
            total += 1.0 - self._plaquette(site_minus_nu, mu, nu)

        return p.beta * total

    def staple_sum(self, site: tuple, mu: int) -> complex:
        """Calcula la suma de staples para un enlace.

        El staple es la contribución de los plaquettes adyacentes
        sin el enlace central. Usado para actualizaciones eficientes.

        Args:
            site: Sitio del enlace
            mu: Dirección

        Returns:
            Suma de staples (como número complejo para U(1))
        """
        p = self.params
        L = p.L
        staple_sum = 0.0 + 0.0j

        for nu in range(p.n_dim):
            if nu == mu:
                continue

            # Staple superior
            site_mu = list(site)
            site_mu[mu] = (site_mu[mu] + 1) % L
            site_mu = tuple(site_mu)

            site_nu = list(site)
            site_nu[nu] = (site_nu[nu] + 1) % L
            site_nu = tuple(site_nu)

            theta_up = (
                self._get_link(site_mu, nu)
                - self._get_link(site_nu, mu)
                - self._get_link(site, nu)
            )
            staple_sum += np.exp(1j * theta_up)

            # Staple inferior
            site_minus_nu = list(site)
            site_minus_nu[nu] = (site_minus_nu[nu] - 1) % L
            site_minus_nu = tuple(site_minus_nu)

            site_mu_minus_nu = list(site_mu)
            site_mu_minus_nu[nu] = (site_mu_minus_nu[nu] - 1) % L
            site_mu_minus_nu = tuple(site_mu_minus_nu)

            theta_down = (
                -self._get_link(site_minus_nu, nu)
                - self._get_link(site_minus_nu, mu)
                + self._get_link(site_mu_minus_nu, nu)
            )
            staple_sum += np.exp(1j * theta_down)

        return staple_sum

    def average_plaquette(self) -> float:
        """Calcula el plaquette promedio.

        ⟨P⟩ = (1/N_plaq) Σ_□ (1/N) Re Tr U_□

        Returns:
            Plaquette promedio
        """
        p = self.params
        total = 0.0
        count = 0

        for idx in np.ndindex(*([p.L] * p.n_dim)):
            for mu in range(p.n_dim):
                for nu in range(mu + 1, p.n_dim):
                    total += self._plaquette(idx, mu, nu)
                    count += 1

        return total / count if count > 0 else 0.0

    def copy(self) -> "WilsonAction":
        """Crea una copia de la configuración."""
        new_action = WilsonAction(self.params)
        new_action.links = self.links.copy()
        return new_action
