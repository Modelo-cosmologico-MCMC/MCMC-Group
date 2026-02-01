"""MCMC-modified Poisson solver for N-body simulations.

MCMC Ontology: The Poisson equation is modified to include:
1. MCV (Masa Cuántica Virtual) contribution to effective density
2. Stratified present effects (S_local varies spatially)
3. Lambda_rel dynamical dark energy contribution

The modified Poisson equation:
    nabla^2 Phi = 4*pi*G * rho_eff(S)

where:
    rho_eff = rho_b + alpha_mcv * rho_lat(S) + rho_Lambda_rel(S)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

from mcmc.core.ontology import G, OMEGA_M, OMEGA_LAMBDA
from mcmc.blocks.block3.config import PoissonModifiedParams


@dataclass
class PoissonState:
    """State of the Poisson solver.

    Attributes:
        potential: Gravitational potential field
        density: Total density field
        density_mcv: MCV contribution to density
        k_grid: Wavenumber grid
    """
    potential: np.ndarray
    density: np.ndarray
    density_mcv: np.ndarray | None
    k_grid: np.ndarray


def rho_lat_simple(S: float, rho_0: float = 1.0) -> float:
    """Simplified latent density rho_lat(S).

    MCV (dark matter equivalent) density decays with S.

    Args:
        S: Entropic coordinate
        rho_0: Normalization density

    Returns:
        Latent density contribution
    """
    # MCV decays as universe evolves toward S_0
    S_star = 48.0  # Peak of structure formation
    sigma = 30.0
    decay = np.exp(-0.5 * ((S - S_star) / sigma) ** 2)
    return rho_0 * OMEGA_M * decay


def lambda_rel_simple(S: float) -> float:
    """Simplified Lambda_rel(S) dynamical dark energy.

    Returns Lambda_rel normalized to present value.

    Args:
        S: Entropic coordinate

    Returns:
        Lambda_rel / Lambda_0
    """
    # Lambda_rel grows as S approaches S_0
    S_trans = 65.0  # Transition to dark energy dominated
    width = 15.0
    transition = 0.5 * (1 + np.tanh((S - S_trans) / width))
    return OMEGA_LAMBDA * transition


class PoissonModified:
    """MCMC-modified Poisson solver using FFT.

    Solves the modified Poisson equation:
        nabla^2 Phi = 4*pi*G * rho_eff

    with MCMC corrections to the effective density.
    """

    def __init__(self, params: PoissonModifiedParams | None = None):
        """Initialize the Poisson solver.

        Args:
            params: Configuration parameters
        """
        self.params = params or PoissonModifiedParams()
        self._setup_grid()

    def _setup_grid(self) -> None:
        """Set up the computational grid and k-space."""
        N = self.params.grid_size
        L = self.params.box_size

        # Physical grid
        self.dx = L / N
        x = np.arange(N) * self.dx
        self.x_grid = x

        # Wavenumber grid
        k = fftfreq(N, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = kx**2 + ky**2 + kz**2

        # Avoid division by zero at k=0
        self.k2[0, 0, 0] = 1.0
        self.k2_safe = self.k2.copy()

    def compute_mcv_density(
        self,
        rho_baryon: np.ndarray,
        S: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute MCV (dark matter) density contribution.

        Args:
            rho_baryon: Baryonic density field
            S: Entropic coordinate (scalar or field)

        Returns:
            MCV density field
        """
        if S is None:
            S = self.params.S_current

        if np.isscalar(S):
            # Uniform S: simple scaling
            rho_lat = rho_lat_simple(S)
            return self.params.alpha_mcv * rho_lat * np.ones_like(rho_baryon)
        else:
            # Stratified S: compute pointwise
            rho_mcv = np.zeros_like(rho_baryon)
            S_arr = np.asarray(S)
            for idx in np.ndindex(rho_baryon.shape):
                rho_mcv[idx] = self.params.alpha_mcv * rho_lat_simple(S_arr[idx])
            return rho_mcv

    def compute_effective_density(
        self,
        rho_baryon: np.ndarray,
        S: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute total effective density for Poisson equation.

        rho_eff = rho_b + alpha_mcv * rho_lat(S)

        Args:
            rho_baryon: Baryonic density field
            S: Entropic coordinate

        Returns:
            Effective density field
        """
        if S is None:
            S = self.params.S_current

        rho_eff = rho_baryon.copy()

        if self.params.include_mcv:
            rho_mcv = self.compute_mcv_density(rho_baryon, S)
            rho_eff = rho_eff + rho_mcv

        return rho_eff

    def solve(
        self,
        rho_baryon: np.ndarray,
        S: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the modified Poisson equation.

        nabla^2 Phi = 4*pi*G * rho_eff

        Args:
            rho_baryon: Baryonic density field
            S: Entropic coordinate

        Returns:
            Gravitational potential Phi
        """
        # Compute effective density with MCMC corrections
        rho_eff = self.compute_effective_density(rho_baryon, S)

        # FFT of density
        rho_k = fftn(rho_eff)

        # Solve in Fourier space: Phi_k = -4*pi*G * rho_k / k^2
        phi_k = -4 * np.pi * G * rho_k / self.k2_safe

        # Zero mean potential (periodic boundary)
        phi_k[0, 0, 0] = 0.0

        # Inverse FFT to get potential
        potential = np.real(ifftn(phi_k))

        return potential

    def compute_acceleration(
        self,
        potential: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gravitational acceleration from potential.

        a = -nabla Phi

        Args:
            potential: Gravitational potential

        Returns:
            Tuple of (ax, ay, az) acceleration components
        """
        # Central difference gradient
        ax = -np.gradient(potential, self.dx, axis=0)
        ay = -np.gradient(potential, self.dx, axis=1)
        az = -np.gradient(potential, self.dx, axis=2)

        return ax, ay, az

    def solve_and_accelerate(
        self,
        rho_baryon: np.ndarray,
        S: float | np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Solve Poisson and compute acceleration in one call.

        Args:
            rho_baryon: Baryonic density field
            S: Entropic coordinate

        Returns:
            Tuple of (potential, (ax, ay, az))
        """
        potential = self.solve(rho_baryon, S)
        acceleration = self.compute_acceleration(potential)
        return potential, acceleration


class PoissonIsolated(PoissonModified):
    """Poisson solver with isolated (vacuum) boundary conditions.

    Uses zero-padding to implement isolated boundaries.
    """

    def __init__(self, params: PoissonModifiedParams | None = None):
        """Initialize with isolated boundary params."""
        if params is None:
            params = PoissonModifiedParams(boundary="isolated")
        super().__init__(params)

    def _setup_grid(self) -> None:
        """Set up padded grid for isolated boundaries."""
        N = self.params.grid_size
        L = self.params.box_size
        N_pad = 2 * N  # Zero-padding for isolated BC

        self.dx = L / N
        self.N_orig = N
        self.N_pad = N_pad

        # Padded wavenumber grid
        k = fftfreq(N_pad, d=self.dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
        self.k2 = kx**2 + ky**2 + kz**2
        self.k2[0, 0, 0] = 1.0
        self.k2_safe = self.k2.copy()

    def solve(
        self,
        rho_baryon: np.ndarray,
        S: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve with isolated boundary conditions.

        Args:
            rho_baryon: Baryonic density field
            S: Entropic coordinate

        Returns:
            Gravitational potential
        """
        N = self.N_orig
        N_pad = self.N_pad

        # Compute effective density
        rho_eff = self.compute_effective_density(rho_baryon, S)

        # Zero-pad the density
        rho_pad = np.zeros((N_pad, N_pad, N_pad))
        rho_pad[:N, :N, :N] = rho_eff

        # Solve in Fourier space
        rho_k = fftn(rho_pad)
        phi_k = -4 * np.pi * G * rho_k / self.k2_safe
        phi_k[0, 0, 0] = 0.0
        phi_pad = np.real(ifftn(phi_k))

        # Extract original region
        return phi_pad[:N, :N, :N]


def create_poisson_solver(
    params: PoissonModifiedParams | None = None,
) -> PoissonModified:
    """Factory function to create appropriate Poisson solver.

    Args:
        params: Configuration parameters

    Returns:
        PoissonModified or PoissonIsolated based on boundary type
    """
    if params is None:
        params = PoissonModifiedParams()

    if params.boundary == "isolated":
        return PoissonIsolated(params)
    else:
        return PoissonModified(params)
